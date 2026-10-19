"""
transformations.py
Seismic data augmentations used to pretrain KenshoDense, and the parameter
grid of augmentation configurations evaluated in the study.

Eight augmentation families are covered:

    GB   Gaussian Blur              (torchvision built-in)
    AS   Amplitude Scaling          (SeismicAmplitudeScaling)
    CJ   Color Jitter               (torchvision built-in)
    SC   Squared Crop               (SquaredCrop)
    VPS  Vertical Patch Swap        (VerticalPatchSwap)
    FF   Frequency Filter           (SeismicFrequencyFilter)
    Tr   Translation                (SeismicTranslation)
    PR   Pseudo Relief              (SeismicPseudoRelief)

Each family is swept over 3 or 4 parameter variants (26 configurations in
total, see ``PARAM_GRID``). Every configuration applies a single
augmentation (after a resize to the working image size), so the effect of
each transformation on KenshoDense can be measured in isolation.

For the Frequency Filter family the swept axis is ``filter_type``
(low_pass / high_pass / mixed) at a fixed (filter_strength, cutoff_freq)
= (0.3, 0.3), which keeps the FF grid the same size as the other
3-variant families.
"""

import random

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from typing import Dict


# ==============================================================================
# TRANSFORM CLASSES
# ==============================================================================

class SeismicAmplitudeScaling:
    """Random amplitude scaling for seismic data (GPU compatible).

    Multiplies the whole tensor by a single factor s ~ U(scale_range)."""

    def __init__(self, scale_range=(0.7, 1.3)):
        self.scale_range = scale_range

    def __call__(self, tensor):
        scale = random.uniform(*self.scale_range)
        return tensor * scale

    def __repr__(self):
        return f"{self.__class__.__name__}(scale_range={self.scale_range})"


class SeismicFrequencyFilter:
    """Simulate frequency filtering effects: low-pass, high-pass, or a
    per-call random choice between the two ("mixed")."""

    def __init__(self, filter_type='mixed', filter_strength=0.5, cutoff_freq=0.3):
        self.filter_type = filter_type
        self.filter_strength = filter_strength
        self.cutoff_freq = cutoff_freq

    def __call__(self, tensor):
        if self.filter_type == 'mixed':
            chosen_filter = random.choice(['low_pass', 'high_pass'])
        else:
            chosen_filter = self.filter_type

        if chosen_filter == 'low_pass':
            return self._apply_low_pass(tensor)
        else:
            return self._apply_high_pass(tensor)

    def _apply_low_pass(self, tensor):
        filtered = tensor.clone()
        kernel_size = max(3, int(1 / self.cutoff_freq))
        if kernel_size % 2 == 0:
            kernel_size += 1

        for _ in range(int(self.filter_strength * 3)):
            if filtered.shape[2] > kernel_size:
                pad = kernel_size // 2
                filtered_h = torch.nn.functional.conv1d(
                    filtered.view(-1, 1, filtered.shape[2]),
                    torch.ones(1, 1, kernel_size, device=tensor.device) / kernel_size,
                    padding=pad
                ).view(filtered.shape)
                filtered = (1 - self.filter_strength) * filtered + self.filter_strength * filtered_h

            if filtered.shape[1] > kernel_size:
                filtered_v = filtered.clone()
                for c in range(filtered.shape[0]):
                    filtered_v[c] = torch.nn.functional.conv1d(
                        filtered[c].unsqueeze(0),
                        torch.ones(1, 1, kernel_size, device=tensor.device) / kernel_size,
                        padding=kernel_size // 2
                    ).squeeze(0)
                filtered = (1 - self.filter_strength) * filtered + self.filter_strength * filtered_v

        return filtered

    def _apply_high_pass(self, tensor):
        low_passed = self._apply_low_pass_simple(tensor)
        high_passed = tensor - low_passed
        return tensor + self.filter_strength * high_passed

    def _apply_low_pass_simple(self, tensor):
        filtered = tensor.clone()
        filtered[:, 1:-1, 1:-1] = (
            tensor[:, 1:-1, 1:-1] +
            tensor[:, :-2, 1:-1] + tensor[:, 2:, 1:-1] +
            tensor[:, 1:-1, :-2] + tensor[:, 1:-1, 2:] +
            tensor[:, :-2, :-2] + tensor[:, :-2, 2:] +
            tensor[:, 2:, :-2] + tensor[:, 2:, 2:]
        ) / 9.0
        return filtered

    def __repr__(self):
        return f"{self.__class__.__name__}(filter_type={self.filter_type}, strength={self.filter_strength}, cutoff={self.cutoff_freq})"


class SeismicPseudoRelief:
    """Pseudo relief: adds a directional (randomly lit) Gaussian-derivative
    gradient to simulate lighting effects. Output is clipped to [0, 1]."""

    def __init__(self, relief_sigma=1.0, relief_intensity=0.3):
        self.relief_sigma = relief_sigma
        self.relief_intensity = relief_intensity

    def __call__(self, tensor):
        if tensor.is_cuda:
            tensor_np = tensor.cpu().numpy()
            device = tensor.device
        else:
            tensor_np = tensor.numpy()
            device = tensor.device

        if tensor_np.shape[0] == 3:
            gray = np.mean(tensor_np, axis=0)
        else:
            gray = tensor_np[0]

        grad_x = gaussian_filter(gray, sigma=self.relief_sigma, order=[0, 1])
        grad_y = gaussian_filter(gray, sigma=self.relief_sigma, order=[1, 0])

        light_angle = random.uniform(0, 2 * np.pi)

        relief = self.relief_intensity * (
            np.cos(light_angle) * grad_x + np.sin(light_angle) * grad_y
        )

        enhanced = tensor_np.copy()
        for c in range(tensor_np.shape[0]):
            enhanced[c] = tensor_np[c] + relief
            enhanced[c] = np.clip(enhanced[c], 0, 1)

        result = torch.from_numpy(enhanced).to(device)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.relief_sigma}, intensity={self.relief_intensity})"


class VerticalPatchSwap:
    """Split the image into horizontal bands of ``patch_height`` rows and
    randomly shuffle their vertical order."""

    def __init__(self, patch_height=50):
        self.patch_height = patch_height

    def __call__(self, tensor):
        c, h, w = tensor.shape
        num_patches = h // self.patch_height + (1 if h % self.patch_height != 0 else 0)

        patches = []
        for i in range(num_patches):
            start_row = i * self.patch_height
            end_row = min((i + 1) * self.patch_height, h)
            patches.append(tensor[:, start_row:end_row, :])

        random.shuffle(patches)
        shuffled = torch.cat(patches, dim=1)

        if shuffled.shape[1] != h:
            shuffled = shuffled[:, :h, :]

        return shuffled

    def __repr__(self):
        return f"{self.__class__.__name__}(patch_height={self.patch_height})"


class SquaredCrop:
    """Random square crop, resized back to the original (H, W).

    ``crop_sizes`` is a list of candidate side lengths; one is drawn per
    call. The study grid passes a single-element list per configuration,
    so each configuration uses one fixed crop size."""

    def __init__(self, crop_sizes=[120, 225]):
        self.crop_sizes = crop_sizes

    def __call__(self, tensor):
        c, h, w = tensor.shape
        crop_size = random.choice(self.crop_sizes)

        crop_size = min(crop_size, min(h, w))

        max_y = h - crop_size
        max_x = w - crop_size

        start_y = random.randint(0, max_y) if max_y > 0 else 0
        start_x = random.randint(0, max_x) if max_x > 0 else 0

        cropped = tensor[:, start_y:start_y + crop_size, start_x:start_x + crop_size]

        resized = torch.nn.functional.interpolate(
            cropped.unsqueeze(0), size=(h, w),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        return resized

    def __repr__(self):
        return f"{self.__class__.__name__}(crop_sizes={self.crop_sizes})"


class SeismicTranslation:
    """Random vertical/horizontal circular (roll) translation by one of
    {-max_translation, 0, +max_translation} pixels on each axis."""

    def __init__(self, max_translation=12):
        self.max_translation = max_translation
        self.translation_options = [-max_translation, 0, max_translation]

    def __call__(self, tensor):
        v_translation = random.choice(self.translation_options)
        h_translation = random.choice(self.translation_options)

        if v_translation != 0:
            tensor = torch.roll(tensor, shifts=v_translation, dims=-2)
        if h_translation != 0:
            tensor = torch.roll(tensor, shifts=h_translation, dims=-1)

        return tensor

    def get_translation_labels(self, tensor):
        v_translation = random.choice(self.translation_options)
        h_translation = random.choice(self.translation_options)
        return v_translation, h_translation

    def __repr__(self):
        return f"{self.__class__.__name__}(max_translation={self.max_translation})"




# ==============================================================================
# PARAMETER GRID
# ==============================================================================
# One entry per (family, parameter variant).
#
# Family : variants
# GB     : k9 sigma(0.1,2.0) / k13 sigma(0.3,2.2) / k11 sigma(0.5,2.5)
# AS     : (0.65,1.35) / (0.7,1.3) / (0.6,1.4)
# CJ     : (0.4,.4,.2,.1) / (0.45,.45,.25,.12) / (0.5,.5,.3,.15)
#          (brightness, contrast, saturation, hue)
# SC     : 90 / 150 / 200 / 225
# VPS    : 25 / 50 / 75 / 100
# FF     : low_pass / high_pass / mixed  @ (strength, cutoff) = (0.3, 0.3)
# Tr     : 8 / 12 / 16
# PR     : (sigma, intensity) = (0.75,0.15) / (1.0,0.2) / (1.5,0.3)

PARAM_GRID = {
    "GB": [
        {"kernel_size": 9, "sigma": (0.1, 2.0)},
        {"kernel_size": 13, "sigma": (0.3, 2.2)},
        {"kernel_size": 11, "sigma": (0.5, 2.5)},
    ],
    "AS": [
        {"scale_range": (0.65, 1.35)},
        {"scale_range": (0.7, 1.3)},
        {"scale_range": (0.6, 1.4)},
    ],
    "CJ": [
        {"params": (0.4, 0.4, 0.2, 0.1)},
        {"params": (0.45, 0.45, 0.25, 0.12)},
        {"params": (0.5, 0.5, 0.3, 0.15)},
    ],
    "SC": [
        {"crop_sizes": [90]},
        {"crop_sizes": [150]},
        {"crop_sizes": [200]},
        {"crop_sizes": [225]},
    ],
    "VPS": [
        {"patch_height": 25},
        {"patch_height": 50},
        {"patch_height": 75},
        {"patch_height": 100},
    ],
    # Sweeps filter_type at a fixed (strength, cutoff). To sweep
    # (strength, cutoff) as well, add the extra combinations here.
    "FF": [
        {"filter_type": "low_pass", "filter_strength": 0.3, "cutoff_freq": 0.3},
        {"filter_type": "high_pass", "filter_strength": 0.3, "cutoff_freq": 0.3},
        {"filter_type": "mixed", "filter_strength": 0.3, "cutoff_freq": 0.3},
    ],
    "Tr": [
        {"max_translation": 8},
        {"max_translation": 12},
        {"max_translation": 16},
    ],
    "PR": [
        {"relief_sigma": 0.75, "relief_intensity": 0.15},
        {"relief_sigma": 1.0, "relief_intensity": 0.2},
        {"relief_sigma": 1.5, "relief_intensity": 0.3},
    ],
}

FAMILIES = tuple(PARAM_GRID.keys())


def _build_transform(family: str, params: dict):
    """Instantiate the augmentation for one (family, params) grid entry."""
    if family == "GB":
        return transforms.GaussianBlur(**params)
    if family == "AS":
        return SeismicAmplitudeScaling(scale_range=params["scale_range"])
    if family == "CJ":
        return transforms.ColorJitter(*params["params"])
    if family == "SC":
        return SquaredCrop(crop_sizes=params["crop_sizes"])
    if family == "VPS":
        return VerticalPatchSwap(patch_height=params["patch_height"])
    if family == "FF":
        return SeismicFrequencyFilter(**params)
    if family == "Tr":
        return SeismicTranslation(max_translation=params["max_translation"])
    if family == "PR":
        return SeismicPseudoRelief(**params)
    raise ValueError(f"Unknown augmentation family: {family!r}")


def get_transform_dict(img_size=(255, 701)) -> Dict[str, "transforms.Compose"]:
    """
    Build the full experiment dict ``{config_key: transforms.Compose}`` from
    ``PARAM_GRID``. 26 configurations in total:
    GB(3) + AS(3) + CJ(3) + SC(4) + VPS(4) + FF(3) + Tr(3) + PR(3) = 26.

    Keys are ``f"{family}_param{i}"`` (1-indexed within the family), e.g.
    ``"GB_param1"``, ``"FF_param2"``.
    """
    base_transforms = [transforms.Resize(img_size)]
    experiments: Dict[str, transforms.Compose] = {}

    for family, variants in PARAM_GRID.items():
        for i, params in enumerate(variants, 1):
            experiments[f"{family}_param{i}"] = transforms.Compose(
                base_transforms + [_build_transform(family, params)]
            )

    return experiments


def describe_grid() -> Dict[str, dict]:
    """Return ``{config_key: params}`` (handy for logging and tables)."""
    return {
        f"{family}_param{i}": params
        for family, variants in PARAM_GRID.items()
        for i, params in enumerate(variants, 1)
    }


if __name__ == "__main__":
    exps = get_transform_dict()
    params = describe_grid()
    print(f"Number of augmentation configurations: {len(exps)}")
    for k in exps:
        print(f"  - {k:<12} {params[k]}")
