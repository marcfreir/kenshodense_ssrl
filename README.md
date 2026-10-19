# KenshoDense

**Dense self-distillation with N views for self-supervised learning on seismic data.**

KenshoDense is a self-supervised learning (SSL) method for pretraining convolutional backbones for dense prediction tasks such as seismic facies segmentation. It combines the negative-free self-distillation of BYOL and TriBYOL with the dense, location-level objective of DenseCL. An online network is trained to predict the representations produced by $N-1$ slowly evolving target networks, each fed a different augmented view of the same image. The prediction is made at two granularities: a global embedding of the whole image and an $S \times S$ grid of local embeddings, where local embeddings are paired across views by backbone-feature similarity. No negative pairs, memory queues or large batches are required: representational collapse is avoided by the asymmetry between the online and target networks (predictor heads exist only on the online side) and by the stop-gradient on the targets.

After pretraining, only the online backbone is kept and used as the encoder of a downstream segmentation network.

---

## Method

### Views

Let $x$ be an input image and $\mathcal{T}$ a distribution of data augmentations. KenshoDense draws $N \ge 2$ independent augmentations $t_0, \dots, t_{N-1} \sim \mathcal{T}$ and produces the views

```math
v_k = t_k(x), \qquad k = 0, \dots, N-1 .
```

View $v_0$ is processed by the online network; each view $v_k$, $k \ge 1$, is processed by its own target network $k$.

### Networks

The **online network**, with parameters $\theta$, consists of a backbone $f_\theta$, a global projector $g_\theta$, a dense projector $h_\theta$, a global predictor $q_\theta$ and a dense predictor $r_\theta$. Each **target network** $k \in \{1, \dots, N-1\}$, with parameters $\xi_k$, has the same backbone and projectors, $f_{\xi_k}$, $g_{\xi_k}$ and $h_{\xi_k}$, but no predictors.

The backbone maps a view to a feature map $F = f(v) \in \mathbb{R}^{C \times H \times W}$. The heads are:

- **Global projector** $g$: global average pooling followed by an MLP,
  Linear$(C, d_h)$ → BN → ReLU → Linear$(d_h, D)$ → BN, giving a vector in $\mathbb{R}^D$.
- **Dense projector** $h$: the same MLP implemented with $1 \times 1$ convolutions, which preserves the spatial layout, followed by adaptive average pooling to an $S \times S$ grid, giving $\mathbb{R}^{D \times S \times S}$.
- **Global predictor** $q$ and **dense predictor** $r$ (online only): Linear$(D, d_p)$ → BN → ReLU → Linear$(d_p, D)$, and its $1 \times 1$-convolution counterpart.

Throughout, $\bar{u} = u / \lVert u \rVert_2$ denotes $\ell_2$ normalisation, $\mathrm{sg}(\cdot)$ the stop-gradient operator, and $s, j \in \{1, \dots, S^2\}$ index the cells of the grid.

### Online and target representations

For the online view,

```math
F^{0} = f_\theta(v_0), \qquad
p = \overline{q_\theta\!\big(g_\theta(F^{0})\big)} \in \mathbb{R}^{D}, \qquad
p^{s} = \overline{r_\theta\!\big(h_\theta(F^{0})\big)_{:,s}} \in \mathbb{R}^{D} .
```

For each target view $k = 1, \dots, N-1$,

```math
F^{k} = f_{\xi_k}(v_k), \qquad
z^{k} = \overline{g_{\xi_k}(F^{k})} \in \mathbb{R}^{D}, \qquad
z^{k,j} = \overline{h_{\xi_k}(F^{k})_{:,j}} \in \mathbb{R}^{D} .
```

### Dense correspondence

Because augmentations displace image content, cell $s$ of the online view and cell $s$ of a target view do not, in general, depict the same structure. Following DenseCL, each online cell is matched to the most similar target cell. Similarity is measured on the backbone features pooled to the same grid,

```math
b^{k,j} = \overline{\mathrm{Pool}_{S}\!\big(F^{k}\big)_{:,j}} \in \mathbb{R}^{C}, \qquad k = 0, \dots, N-1,
```

and the correspondence between the online view and target view $k$ is

```math
c_k(s) = \arg\max_{j \in \{1, \dots, S^2\}} \big\langle b^{0,s},\, b^{k,j} \big\rangle .
```

### Objective

Both terms of the objective use the normalised regression loss of BYOL, $2 - 2\langle \bar{a}, \bar{b} \rangle = \lVert \bar{a} - \bar{b} \rVert_2^2$.

The **global loss** with respect to target $k$ is

```math
\mathcal{L}^{k}_{\mathrm{global}} = 2 - 2 \big\langle p,\, \mathrm{sg}(z^{k}) \big\rangle ,
```

and the **dense loss** with respect to target $k$, averaged over the grid, is

```math
\mathcal{L}^{k}_{\mathrm{dense}} = \frac{1}{S^{2}} \sum_{s=1}^{S^{2}} \Big( 2 - 2 \big\langle p^{s},\, \mathrm{sg}\big(z^{k,\,c_k(s)}\big) \big\rangle \Big) .
```

The two are combined with a weight $\lambda \in [0, 1]$ and averaged over the $N-1$ targets:

```math
\mathcal{L}^{k} = (1 - \lambda)\, \mathcal{L}^{k}_{\mathrm{global}} + \lambda\, \mathcal{L}^{k}_{\mathrm{dense}},
\qquad
\mathcal{L} = \frac{1}{N-1} \sum_{k=1}^{N-1} \mathcal{L}^{k} .
```

All terms are additionally averaged over the mini-batch. The loss is minimised with respect to $\theta$ only; view $v_0$ is always the online view, so the loss is not symmetrised.

### Target update

The target networks receive no gradients. After each optimisation step, every target is updated as an exponential moving average (EMA) of the online backbone and projectors:

```math
\xi_k \leftarrow \tau_t\, \xi_k + (1 - \tau_t)\, \theta, \qquad k = 1, \dots, N-1 ,
```

where the momentum $\tau_t$ increases from $\tau_0$ to $1$ following a cosine schedule over the $T$ training steps:

```math
\tau_t = \tau_0 + (1 - \tau_0)\, \frac{1 - \cos\!\big(\pi \min(t, T) / T\big)}{2} .
```

All target networks are initialised as copies of the online network. With this update they therefore keep identical parameters during training and differ only in the view they receive. A TriBYOL-style alternative, in which a single target is updated per step in round-robin order (so the targets diverge), is also implemented.

### Optimisation

The online parameters are optimised with AdamW. The learning rate follows a linear warm-up of $W$ steps and a cosine decay:

```math
\eta_t =
\begin{cases}
\eta_0 \, t / W, & t < W, \\[1mm]
\dfrac{\eta_0}{2} \Big( 1 + \cos\!\Big( \pi \min\!\Big( \dfrac{t - W}{T - W},\, 1 \Big) \Big) \Big), & t \ge W .
\end{cases}
```

---

## Algorithm

```text
Algorithm 1  KenshoDense pretraining
─────────────────────────────────────────────────────────────────────────────
Require: unlabeled dataset X, augmentation distribution T, number of views N,
         grid size S, loss weight λ, initial EMA momentum τ₀,
         base learning rate η₀, warm-up steps W, total steps T_max
 1: initialise online parameters θ = (f_θ, g_θ, h_θ, q_θ, r_θ)
 2: for k = 1, …, N−1 do
 3:     ξ_k ← (f_θ, g_θ, h_θ)                          ▷ target copies, no predictors
 4: end for
 5: for t = 0, …, T_max − 1 do
 6:     sample a mini-batch {x_i} from X
 7:     for k = 0, …, N−1 do
 8:         v_k ← t_k(x), t_k ~ T                         ▷ independent views
 9:     end for
10:     F⁰ ← f_θ(v₀)
11:     p  ← normalise(q_θ(g_θ(F⁰)))                      ▷ global prediction
12:     pˢ ← normalise(r_θ(h_θ(F⁰))_s),  s = 1, …, S²     ▷ dense predictions
13:     b⁰ˢ ← normalise(Pool_S(F⁰)_s)
14:     L ← 0
15:     for k = 1, …, N−1 do                              ▷ no gradient through targets
16:         Fᵏ  ← f_ξk(v_k)
17:         zᵏ  ← normalise(g_ξk(Fᵏ))
18:         zᵏʲ ← normalise(h_ξk(Fᵏ)_j),  j = 1, …, S²
19:         bᵏʲ ← normalise(Pool_S(Fᵏ)_j)
20:         c_k(s) ← argmax_j ⟨b⁰ˢ, bᵏʲ⟩,  s = 1, …, S²    ▷ dense correspondence
21:         L_global ← 2 − 2 ⟨p, sg(zᵏ)⟩
22:         L_dense  ← (1/S²) Σ_s [ 2 − 2 ⟨pˢ, sg(zᵏ,c_k(s))⟩ ]
23:         L ← L + (1 − λ) L_global + λ L_dense
24:     end for
25:     L ← L / (N − 1)
26:     θ ← AdamW(θ, ∇_θ L, η_t)                          ▷ warm-up + cosine η_t
27:     τ_t ← τ₀ + (1 − τ₀)(1 − cos(π t / T_max)) / 2
28:     for k = 1, …, N−1 do
29:         ξ_k ← τ_t ξ_k + (1 − τ_t) θ                    ▷ EMA update
30:     end for
31: end for
32: return f_θ                                            ▷ online backbone
```

---

## Default hyperparameters

| Symbol | Description | Value |
|---|---|---|
| $f$ | backbone | ResNet-50 (width ×1, ×2 or ×4), dilated last two stages (output stride 8) |
| $C$ | backbone channels | $2048 \times$ width |
| $N$ | number of views | 3 |
| $S$ | dense grid size | 7 |
| $D$ | embedding dimension | 512 |
| $d_h$ | projector hidden dimension | 4096 |
| $d_p$ | predictor hidden dimension | 1024 |
| $\lambda$ | dense loss weight | 0.5 |
| $\tau_0$ | initial EMA momentum | 0.996 |
| $\eta_0$ | base learning rate | $10^{-3}$ |
| — | optimiser | AdamW, $\beta = (0.9, 0.95)$, $\epsilon = 10^{-6}$, weight decay $10^{-4}$ |
| $W$ | warm-up steps | 100 |
| — | batch size / epochs | 16 / 100 |

---

## References

- J.-B. Grill et al. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* NeurIPS 2020.
- X. Wang, R. Zhang, C. Shen, T. Kong, L. Li. *Dense Contrastive Learning for Self-Supervised Visual Pre-Training.* CVPR 2021.
- G. Li, R. Togo, T. Ogawa, M. Haseyama. *TriBYOL: Triplet BYOL for Self-Supervised Representation Learning.* ICASSP 2022.
- T. Chen, S. Kornblith, M. Norouzi, G. Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020.
- L.-C. Chen, G. Papandreou, F. Schroff, H. Adam. *Rethinking Atrous Convolution for Semantic Image Segmentation.* arXiv:1706.05587, 2017.

The DeepLabV3 implementation and the training pipeline build on the [Minerva](https://github.com/discovery-unicamp/Minerva) project. We thank all of its authors.