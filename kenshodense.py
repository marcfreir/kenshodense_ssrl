"""
kenshodense.py — KenshoDense: Dense Self-Distillation with N Views
====================================================================
A self-supervised learning (SSL) model that combines the dense spatial
representation learning of DenseCL with the negative-pair-free
self-distillation of BYOL / TriBYOL.

Augmentation contract
---------------------
The model generates its own ``num_views`` augmented views internally: the
batch delivered by the DataLoader contains RAW images (``batch[0]`` or
``batch`` itself), and KenshoDense calls ``self.augmentations``
independently ``num_views`` times per image.

Design principles
-----------------
1. **No negative pairs** — collapse is prevented by an asymmetric
   online → target architecture (predictor on the online branch only)
   together with stop-gradient on all target branches, as in BYOL.

2. **Dense + global loss** — like DenseCL, each view produces both a
   global embedding and a dense S×S grid of local embeddings. The loss
   is computed at both granularities and combined with weight λ.

3. **N views / N−1 target networks** — the user chooses ``num_views ≥ 2``.
   There is exactly one online network and ``num_views - 1`` EMA target
   networks (one per non-online view). Targets are updated with a
   cosine-annealed EMA coefficient τ. By default every target is updated
   at every step (``_ema_update_all_targets``); a TriBYOL-style
   round-robin update (``_ema_update_alternating``) is also provided.

4. **Dense correspondence** — the backbone-feature argmax matching from
   DenseCL (§3.3, Eq. 3) finds the positive local-feature pair for each
   target view, keeping the spatial alignment signal.

5. **Flexible optimizer** — priority system:
   factory > class > named preset (sgd | adam | adamw | lars),
   with AdamW as the default.

Architecture
------------
  Online branch (gradients flow):
    backbone_q  →  F_q  (N, C, H, W)
                 ↓              ↓
         global_proj_q      dense_proj_q
              ↓                   ↓
         global_pred_q       dense_pred_q     ← predictors (online only)
              ↓                   ↓
            p_g (N, D)        p_d (N, D, S²)

  Target branch i  (stop-gradient, EMA-updated):
    backbone_t[i] →  F_t  (N, C, H, W)
                  ↓              ↓
         global_proj_t[i]   dense_proj_t[i]
              ↓                   ↓
            z_g (N, D)        z_d (N, D, S²)   ← no predictor

Loss (per target i, then averaged over all N-1 targets):
  L_global_i = 2 − 2 · cosine_sim(p_g, z_g_i)
  L_dense_i  = (1/S²) Σ_s [ 2 − 2 · cosine_sim(p_d^s, z_d_i^{c(s)}) ]
                where c(s) = argmax_j sim(F_q^s, F_t^j)
  L_i        = (1 − λ) · L_global_i + λ · L_dense_i
  L          = (1 / (N−1)) Σ_i L_i

See README.md for the full formulation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
import math

import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.optim import Optimizer
import lightning as L


# ---------------------------------------------------------------------------
# Shared projection / predictor heads
# ---------------------------------------------------------------------------

class GlobalProjectionHead(nn.Module):
    """GAP → Linear(C, hidden) → BN → ReLU → Linear(hidden, D).

    Follows the BYOL projection head convention (BatchNorm inside) rather
    than the plain MoCo-v2 head, since we no longer use a queue and BN
    in the projector is standard for distillation methods.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 512):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (N, C, H, W) → (N, D)."""
        return self.mlp(self.gap(feat).flatten(1))


class GlobalPredictorHead(nn.Module):
    """Online-only predictor: Linear(D, hidden) → BN → ReLU → Linear(hidden, D).

    The asymmetry between online (has predictor) and target (no predictor)
    is the key mechanism preventing representational collapse in BYOL-style
    methods.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) → (N, D)."""
        return self.mlp(x)


class DenseProjectionHead(nn.Module):
    """1×1 conv → BN → ReLU → 1×1 conv (spatially preserving).

    Produces dense feature vectors of shape (N, D, H, W).
    BN on the dense head follows the SimSiam / DenseCL convention.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 4096, output_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim, affine=False),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (N, C, H, W) → (N, D, H, W)."""
        return self.conv(feat)


class DensePredictorHead(nn.Module):
    """Online-only dense predictor: 1×1 conv → BN → ReLU → 1×1 conv.

    Applied after the dense projection head on the online branch only.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D, H, W) → (N, D, H, W)."""
        return self.conv(x)


# ---------------------------------------------------------------------------
# KenshoDense Lightning module
# ---------------------------------------------------------------------------

class KenshoDense(L.LightningModule):
    """Dense self-distillation SSL with N flexible, internally-generated views.

    Augmentation contract
    ----------------------
    Augmentations happen INSIDE the model. The pretrain script hands over
    batches of raw images — no multi-view dataset wrapping is required. Each training/validation/test step calls
    ``self.augmentations`` independently ``num_views`` times to build the
    online view (view 0) and the ``num_views - 1`` target views.

    Parameters
    ----------
    backbone : nn.Module
        Convolutional backbone returning spatial feature maps (N, C, H, W).
    backbone_out_dim : int
        Number of output channels C from the backbone. Default 2048.
        Ignored when the backbone exposes an ``out_channels`` attribute
        (e.g. DeepLabV3Backbone with width_multiplier=2 -> 4096), so the
        heads always match the backbone width (RN50 1x / 2x / 4x).
    proj_hidden_dim : int
        Hidden dimension inside projection heads. Default 4096.
    pred_hidden_dim : int
        Hidden dimension inside predictor heads. Default 1024.
    output_dim : int
        Final embedding dimension D for both global and dense heads.
        Default 512.
    grid_size : int
        Spatial grid size S.  Backbone features are adaptively pooled to
        S×S before the dense correspondence step.  Default 7.
    num_views : int
        Total number of augmented views per image (≥ 2).
        View 0  → online branch.
        Views 1 … N-1  → one target network each.
        Default 3.
    base_target_ema : float
        Starting value of the EMA coefficient τ₀.  τ is annealed to 1.0
        with a cosine schedule over training (same as BYOL / TriBYOL).
        Default 0.996.
    loss_lambda : float
        Weight λ balancing global vs dense distillation loss.
        L = (1−λ)·L_global + λ·L_dense.  Default 0.5.
    learning_rate : float
        Base learning rate. Default 0.03.
    momentum : float
        SGD momentum (also used as default for LARS). Default 0.9.
    weight_decay : float
        Weight decay. Default 0.0004.
    max_epochs : int
        Total training epochs. Default 100.
    steps_per_epoch : int
        Steps per epoch, used with max_epochs for both the τ schedule and
        the LR scheduler (``max_steps = max_epochs * steps_per_epoch``).
        Default 100.
    batch_size : int
        Effective batch size. Used for LR scaling. Default 128.
    augmentations : Callable, optional
        Augmentation pipeline applied independently to build each of the
        ``num_views`` views. If None, all views fall back to the raw image.
    optimizer_name : str
        Preset: "sgd" | "adam" | "adamw" | "lars". Default "adamw".
    optimizer_class : type, optional
        Custom optimizer class; overrides optimizer_name.
    optimizer_params : dict, optional
        Extra kwargs merged into the chosen optimizer's defaults.
    optimizer_factory : callable, optional
        Highest-priority override: ``factory(params, lr) → Optimizer``.
    use_scheduler : bool
        Attach cosine-annealing + linear-warmup LR scheduler. Default True.
    warmup_steps : int
        Linear warmup duration in steps. Default 100.
    scale_lr_with_batch_size : bool
        Scale lr linearly as ``lr × batch_size / lr_scaling_base``. Default False.
    lr_scaling_base : int
        Reference batch size for LR scaling. Default 256.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_dim: int = 2048,
        proj_hidden_dim: int = 4096,
        pred_hidden_dim: int = 1024,
        output_dim: int = 512,
        grid_size: int = 7,
        num_views: int = 3,
        base_target_ema: float = 0.996,
        loss_lambda: float = 0.5,
        # ── Common hyperparameters ────────────────────────────────────────
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 0.0004,
        max_epochs: int = 100,
        steps_per_epoch: int = 100,
        batch_size: int = 128,
        augmentations: Optional[Callable] = None,
        # ── Optimizer ────────────────────────────────────────────────────
        optimizer_name: str = "adamw",
        optimizer_class: Optional[type] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        optimizer_factory: Optional[Callable] = None,
        # ── LR scheduler ─────────────────────────────────────────────────
        use_scheduler: bool = True,
        warmup_steps: int = 100,
        scale_lr_with_batch_size: bool = False,
        lr_scaling_base: int = 256,
    ):
        super().__init__()

        # Follow the backbone's real output width (2048 * width_multiplier
        # for DeepLabV3Backbone) so 1x / 2x / 4x backbones just work.
        backbone_out_dim = getattr(backbone, "out_channels", backbone_out_dim)
        self.backbone_out_dim = backbone_out_dim

        if num_views < 2:
            raise ValueError(f"num_views must be ≥ 2, got {num_views}.")

        self.num_views = num_views
        self.num_targets = num_views - 1   # one target network per non-online view
        self.base_target_ema = base_target_ema
        self.loss_lambda = loss_lambda
        self.grid_size = grid_size

        # ── Common hyperparameters ─────────────────────────────────────────
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.augmentations = augmentations

        # ── Optimizer / scheduler config ──────────────────────────────────
        self.optimizer_name = optimizer_name
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params or {}
        self.optimizer_factory = optimizer_factory
        self.use_scheduler = use_scheduler
        self.warmup_steps = warmup_steps
        self.scale_lr_with_batch_size = scale_lr_with_batch_size
        self.lr_scaling_base = lr_scaling_base

        # Total steps for both the τ schedule and the LR scheduler.
        self.max_steps = max_epochs * steps_per_epoch

        # ── Online backbone ───────────────────────────────────────────────
        self.backbone_online = backbone

        # ── Online projection heads ───────────────────────────────────────
        self.global_proj_online = GlobalProjectionHead(
            backbone_out_dim, proj_hidden_dim, output_dim
        )
        self.dense_proj_online = DenseProjectionHead(
            backbone_out_dim, proj_hidden_dim, output_dim
        )

        # ── Online predictor heads (asymmetry → prevents collapse) ────────
        self.global_pred_online = GlobalPredictorHead(
            output_dim, pred_hidden_dim, output_dim
        )
        self.dense_pred_online = DensePredictorHead(
            output_dim, pred_hidden_dim, output_dim
        )

        # ── Target networks — one per non-online view ─────────────────────
        # Each target has its own backbone + global proj + dense proj.
        # No predictor on target branches (asymmetry is the key).
        self.backbone_targets   = nn.ModuleList()
        self.global_proj_targets = nn.ModuleList()
        self.dense_proj_targets  = nn.ModuleList()

        for _ in range(self.num_targets):
            self.backbone_targets.append(copy.deepcopy(self.backbone_online))
            self.global_proj_targets.append(copy.deepcopy(self.global_proj_online))
            self.dense_proj_targets.append(copy.deepcopy(self.dense_proj_online))

        # Freeze all target parameters — EMA updates only
        for t_idx in range(self.num_targets):
            self._set_requires_grad(self.backbone_targets[t_idx], False)
            self._set_requires_grad(self.global_proj_targets[t_idx], False)
            self._set_requires_grad(self.dense_proj_targets[t_idx], False)

        # ── Adaptive pool: backbone features → S×S grid ───────────────────
        self.pool_to_grid = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # ── Alternating EMA update tracker ────────────────────────────────
        # Used only by _ema_update_alternating (TriBYOL-style round-robin,
        # one target per step). The default training step updates all
        # targets every step via _ema_update_all_targets.
        self._ema_step = 0

        # Loss history for external inspection / plotting.
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool):
        for p in module.parameters():
            p.requires_grad = value

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def _online_params(self) -> List[nn.Parameter]:
        """All parameters that receive gradient updates."""
        return (
            list(self.backbone_online.parameters())
            + list(self.global_proj_online.parameters())
            + list(self.dense_proj_online.parameters())
            + list(self.global_pred_online.parameters())
            + list(self.dense_pred_online.parameters())
        )

    # ------------------------------------------------------------------
    # EMA coefficient schedule  (cosine, τ₀ → 1.0)
    # ------------------------------------------------------------------

    def _current_tau(self) -> float:
        """τ = τ₀ + (1 − τ₀) · (1 − cos(π · t/T)) / 2"""
        step = min(self.global_step, self.max_steps)
        progress = step / max(self.max_steps, 1)
        return self.base_target_ema + (1.0 - self.base_target_ema) * (
            1.0 - math.cos(math.pi * progress)
        ) / 2.0

    # ------------------------------------------------------------------
    # EMA updates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ema_update_target(self, t_idx: int, tau: float):
        """Update a single target network via EMA from the online network."""
        pairs = [
            (self.backbone_online,    self.backbone_targets[t_idx]),
            (self.global_proj_online, self.global_proj_targets[t_idx]),
            (self.dense_proj_online,  self.dense_proj_targets[t_idx]),
        ]
        for online_m, target_m in pairs:
            for p_o, p_t in zip(online_m.parameters(), target_m.parameters()):
                p_t.data = tau * p_t.data + (1.0 - tau) * p_o.data

    @torch.no_grad()
    def _ema_update_all_targets(self, tau: float):
        """Update every target network at each step (simpler default)."""
        for t_idx in range(self.num_targets):
            self._ema_update_target(t_idx, tau)

    @torch.no_grad()
    def _ema_update_alternating(self, tau: float):
        """Update one target at a time, cycling round-robin (TriBYOL style)."""
        t_idx = self._ema_step % self.num_targets
        self._ema_update_target(t_idx, tau)
        self._ema_step += 1

    # ------------------------------------------------------------------
    # Forward: online branch
    # ------------------------------------------------------------------

    def _encode_online(self, x: torch.Tensor) -> Tuple[
        torch.Tensor,   # p_global : (N, D)      — after predictor
        torch.Tensor,   # p_dense  : (N, D, S²)  — after predictor, flattened
        torch.Tensor,   # f_bb     : (N, C, S²)  — ℓ2-normed backbone features
    ]:
        feat = self.backbone_online(x)                       # (N, C, H, W)

        # Global branch
        z_global = self.global_proj_online(feat)             # (N, D)
        p_global = self.global_pred_online(z_global)         # (N, D)
        p_global = normalize(p_global, dim=1)

        # Dense branch
        z_dense_map = self.dense_proj_online(feat)           # (N, D, H, W)
        if z_dense_map.shape[-1] != self.grid_size:
            z_dense_map = self.pool_to_grid(z_dense_map)     # (N, D, S, S)
        p_dense_map = self.dense_pred_online(z_dense_map)    # (N, D, S, S)
        p_dense = normalize(p_dense_map, dim=1).flatten(2)   # (N, D, S²)

        # Backbone features pooled to S×S for correspondence matching
        f_bb = self.pool_to_grid(feat).flatten(2)            # (N, C, S²)
        f_bb = normalize(f_bb, dim=1)

        return p_global, p_dense, f_bb

    # ------------------------------------------------------------------
    # Forward: target branch i  (stop-gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_target(self, x: torch.Tensor, t_idx: int) -> Tuple[
        torch.Tensor,   # z_global : (N, D)
        torch.Tensor,   # z_dense  : (N, D, S²)
        torch.Tensor,   # f_bb     : (N, C, S²)
    ]:
        feat = self.backbone_targets[t_idx](x)               # (N, C, H, W)

        z_global = self.global_proj_targets[t_idx](feat)     # (N, D)
        z_global = normalize(z_global, dim=1)

        z_dense_map = self.dense_proj_targets[t_idx](feat)   # (N, D, H, W)
        if z_dense_map.shape[-1] != self.grid_size:
            z_dense_map = self.pool_to_grid(z_dense_map)
        z_dense = normalize(z_dense_map, dim=1).flatten(2)   # (N, D, S²)

        f_bb = self.pool_to_grid(feat).flatten(2)            # (N, C, S²)
        f_bb = normalize(f_bb, dim=1)

        return z_global, z_dense, f_bb

    # ------------------------------------------------------------------
    # Dense correspondence  (identical logic to DenseCL §3.3, Eq. 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_correspondence(
        f_q: torch.Tensor,   # (N, C, S²)  ℓ2-normed online backbone features
        f_t: torch.Tensor,   # (N, C, S²)  ℓ2-normed target backbone features
    ) -> torch.Tensor:
        """Return index tensor c of shape (N, S²).

        c[n, i] = argmax_j cosine_sim(f_q[n, :, i], f_t[n, :, j])
        """
        sim = torch.bmm(f_q.permute(0, 2, 1), f_t)   # (N, S², S²)
        return sim.argmax(dim=2)                        # (N, S²)

    # ------------------------------------------------------------------
    # Per-pair loss functions  (cosine distillation, no negatives)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """BYOL-style loss: 2 − 2 · mean cosine_sim(p, z).

        Both p and z must already be ℓ2-normalised.
        For global: p, z ∈ (N, D).
        For dense:  p, z ∈ (N·S², D).
        """
        return 2.0 - 2.0 * (p * z).sum(dim=1).mean()

    def _pair_loss(
        self,
        p_global: torch.Tensor,   # (N, D)
        p_dense:  torch.Tensor,   # (N, D, S²)
        z_global: torch.Tensor,   # (N, D)
        z_dense:  torch.Tensor,   # (N, D, S²)
        f_q:      torch.Tensor,   # (N, C, S²)  online backbone features
        f_t:      torch.Tensor,   # (N, C, S²)  target backbone features
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined loss for one (online, target_i) pair.

        Returns
        -------
        loss        : scalar  weighted combination
        loss_global : scalar  global distillation component
        loss_dense  : scalar  dense distillation component
        """
        # ── Global loss ───────────────────────────────────────────────────
        loss_global = self._cosine_loss(p_global, z_global.detach())

        # ── Dense correspondence & loss ───────────────────────────────────
        corr_idx = self._find_correspondence(f_q, f_t)          # (N, S²)

        # Gather target dense embeddings at matched positions
        corr_idx_exp = corr_idx.unsqueeze(1).expand(            # (N, D, S²)
            -1, z_dense.size(1), -1
        )
        z_dense_matched = torch.gather(z_dense.detach(), 2, corr_idx_exp)  # (N, D, S²)

        # Flatten to (N·S², D) for the cosine loss
        N, D, S2 = p_dense.shape
        p_flat = p_dense.permute(0, 2, 1).reshape(N * S2, D)   # (N·S², D)
        z_flat = z_dense_matched.permute(0, 2, 1).reshape(N * S2, D)

        loss_dense = self._cosine_loss(p_flat, z_flat)

        loss = (1.0 - self.loss_lambda) * loss_global + self.loss_lambda * loss_dense
        return loss, loss_global, loss_dense

    # ------------------------------------------------------------------
    # View generation (internal augmentation)
    # ------------------------------------------------------------------

    def _make_views(self, batch: Any, augment: bool) -> List[torch.Tensor]:
        """Build ``num_views`` views from a batch of raw images.

        Parameters
        ----------
        batch : Any
            Batch as delivered by the DataLoader. Expected to be
            ``(images, label)`` or a bare image tensor.
        augment : bool
            If True, each view is generated by an independent call to
            ``self.augmentations``. If False (used for validation/test),
            all views are identical copies of the raw image.
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            images = batch[0]
        else:
            images = batch

        if augment and self.augmentations is not None:
            views = [
                torch.stack([self.augmentations(img) for img in images])
                for _ in range(self.num_views)
            ]
        else:
            views = [images for _ in range(self.num_views)]

        return [v.to(self.device) for v in views]

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------

    def _shared_step(self, batch: Any, augment: bool, update_targets: bool = False):
        """Core forward + loss computation shared across train/val/test.

        Parameters
        ----------
        augment : bool
            Whether views are generated via ``self.augmentations`` (train)
            or are identical unaugmented copies of the raw image
            (validation/test).
        update_targets : bool
            If True, apply EMA update to all target networks after computing
            the loss.  Should be True only during training.
        """
        views = self._make_views(batch, augment=augment)   # list of num_views tensors

        # Online branch: view[0] is always the online input
        p_global, p_dense, f_q = self._encode_online(views[0])

        # Accumulate loss across all target branches
        total_loss        = torch.tensor(0.0, device=self.device)
        total_loss_global = torch.tensor(0.0, device=self.device)
        total_loss_dense  = torch.tensor(0.0, device=self.device)

        for t_idx in range(self.num_targets):
            # Target branch t_idx uses view t_idx+1
            z_global, z_dense, f_t = self._encode_target(views[t_idx + 1], t_idx)

            loss_i, loss_g_i, loss_d_i = self._pair_loss(
                p_global, p_dense, z_global, z_dense, f_q, f_t
            )
            total_loss        = total_loss        + loss_i
            total_loss_global = total_loss_global + loss_g_i
            total_loss_dense  = total_loss_dense  + loss_d_i

        # Average over N-1 pairs
        total_loss        = total_loss        / self.num_targets
        total_loss_global = total_loss_global / self.num_targets
        total_loss_dense  = total_loss_dense  / self.num_targets

        if update_targets:
            tau = self._current_tau()
            self._ema_update_all_targets(tau)

        return total_loss, total_loss_global, total_loss_dense

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, loss_global, loss_dense = self._shared_step(
            batch, augment=True, update_targets=True
        )

        self.log("train_loss",        loss,        prog_bar=True,  sync_dist=True, on_step=True,  on_epoch=True)
        self.log("train_loss_global", loss_global, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("train_loss_dense",  loss_dense,  prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("tau", self._current_tau(),        prog_bar=False, sync_dist=True, on_step=True,  on_epoch=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Validation uses identical (unaugmented) views.
        loss, loss_global, loss_dense = self._shared_step(
            batch, augment=False, update_targets=False
        )

        self.log("val_loss",        loss,        prog_bar=True,  sync_dist=True, on_step=False, on_epoch=True)
        self.log("val_loss_global", loss_global, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val_loss_dense",  loss_dense,  prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, loss_global, loss_dense = self._shared_step(
            batch, augment=False, update_targets=False
        )

        self.log("test_loss",        loss,        prog_bar=True,  sync_dist=True, on_step=False, on_epoch=True)
        self.log("test_loss_global", loss_global, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("test_loss_dense",  loss_dense,  prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Epoch hooks — loss history tracking
    # ------------------------------------------------------------------

    def on_train_epoch_end(self):
        metrics = self.trainer.logged_metrics
        if "train_loss_epoch" in metrics:
            self.train_losses.append(float(metrics["train_loss_epoch"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.logged_metrics
        if "val_loss" in metrics:
            self.val_losses.append(float(metrics["val_loss"]))

    def get_loss_history(self) -> Dict[str, List[float]]:
        return {
            "train_losses": self.train_losses.copy(),
            "val_losses": self.val_losses.copy(),
        }

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure optimizer with optional cosine-annealing + warmup scheduler.

        Priority order (highest first):
          1. optimizer_factory  — custom callable
          2. optimizer_class    — custom class
          3. optimizer_name     — preset: "sgd" | "adam" | "adamw" | "lars"

        Only online parameters receive gradient updates; target networks
        are updated exclusively via EMA.
        """
        lr = self.learning_rate
        if self.scale_lr_with_batch_size:
            lr = lr * self.batch_size / self.lr_scaling_base

        optimizer = self._create_optimizer(lr)

        if not self.use_scheduler:
            return optimizer

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            progress = (step - self.warmup_steps) / max(
                self.max_steps - self.warmup_steps, 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _create_optimizer(self, lr: float) -> Optimizer:
        if self.optimizer_factory is not None:
            return self.optimizer_factory(self._online_params(), lr)

        if self.optimizer_class is not None:
            params = {"lr": lr, **self.optimizer_params}
            return self.optimizer_class(self._online_params(), **params)

        if self.optimizer_name == "sgd":
            return self._create_sgd_optimizer(lr)
        elif self.optimizer_name == "adam":
            return self._create_adam_optimizer(lr)
        elif self.optimizer_name == "adamw":
            return self._create_adamw_optimizer(lr)
        elif self.optimizer_name == "lars":
            return self._create_lars_optimizer(lr)
        else:
            raise ValueError(
                f"Unknown optimizer_name={self.optimizer_name!r}. "
                "Choose from 'sgd', 'adam', 'adamw', 'lars', or supply "
                "optimizer_class / optimizer_factory."
            )

    def _create_sgd_optimizer(self, lr: float) -> torch.optim.SGD:
        defaults = {"momentum": self.momentum, "weight_decay": self.weight_decay}
        params = {**defaults, **self.optimizer_params}
        return torch.optim.SGD(self._online_params(), lr=lr, **params)

    def _create_adam_optimizer(self, lr: float) -> torch.optim.Adam:
        defaults = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": self.weight_decay}
        params = {**defaults, **self.optimizer_params}
        return torch.optim.Adam(self._online_params(), lr=lr, **params)

    def _create_adamw_optimizer(self, lr: float) -> torch.optim.AdamW:
        defaults = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": self.weight_decay}
        params = {**defaults, **self.optimizer_params}
        return torch.optim.AdamW(self._online_params(), lr=lr, **params)

    def _create_lars_optimizer(self, lr: float) -> Optimizer:
        try:
            from lars import LARS
        except ImportError as exc:
            raise ImportError(
                "LARS optimizer not found. Install it or make lars.py available "
                "in the Python path."
            ) from exc
        defaults = {
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "trust_coef": 1e-3,
            "exclude_bias_and_bn": True,
        }
        params = {**defaults, **self.optimizer_params}
        return LARS(self._online_params(), lr=lr, **params)

    # ------------------------------------------------------------------
    # Accessor for downstream fine-tuning
    # ------------------------------------------------------------------

    def get_encoder(self) -> nn.Module:
        """Return the online backbone for downstream fine-tuning."""
        return self.backbone_online

