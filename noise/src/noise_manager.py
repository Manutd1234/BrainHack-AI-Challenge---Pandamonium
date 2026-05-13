"""Manages adversarial noise generation with a YOLO26 surrogate.

The primary attack is PGD against the same YOLO26 family used by the CV model.
The implementation keeps the perturbation within a conservative pixel budget
and falls back to lightweight texture/color perturbations if the surrogate is
not available in a local test environment.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Iterable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_torch_available = False
_surrogate_model = None
_surrogate_kind = None

try:
    import torch
    import torch.nn.functional as F

    _torch_available = True
except ImportError:
    pass


def _candidate_surrogates() -> list[str]:
    configured = os.getenv("YOLO_SURROGATE_PATH")
    candidates = [
        configured,
        "models/best.pt",
        "best.pt",
        "models/yolo26l.pt",
        "models/yolo26m.pt",
        "yolo26l.pt",
        "yolo26m.pt",
        "yolo26s.pt",
    ]
    return [path for path in candidates if path]


def _get_surrogate_model():
    """Lazy-load the YOLO26 white-box surrogate, with MobileNet fallback."""
    global _surrogate_model, _surrogate_kind
    if _surrogate_model is not None or not _torch_available:
        return _surrogate_model, _surrogate_kind

    for model_path in _candidate_surrogates():
        try:
            from ultralytics import YOLO

            if not os.path.exists(model_path) and not model_path.startswith("yolo26"):
                continue
            yolo = YOLO(model_path)
            _surrogate_model = yolo.model.eval()
            _surrogate_kind = "yolo26"
            if torch.cuda.is_available():
                _surrogate_model = _surrogate_model.cuda()
            logger.info("Loaded YOLO26 surrogate model: %s", model_path)
            return _surrogate_model, _surrogate_kind
        except Exception as exc:
            logger.warning("Could not load YOLO26 surrogate %s: %s", model_path, exc)

    try:
        import torchvision.models as models

        _surrogate_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        ).eval()
        _surrogate_kind = "mobilenet"
        if torch.cuda.is_available():
            _surrogate_model = _surrogate_model.cuda()
        logger.info("Loaded MobileNetV2 fallback surrogate.")
    except Exception as exc:
        logger.warning("Failed to load fallback surrogate model: %s", exc)

    return _surrogate_model, _surrogate_kind


def _iter_tensors(value: Any) -> Iterable[Any]:
    if _torch_available and torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _yolo_confidence(outputs: Any):
    tensors = [tensor for tensor in _iter_tensors(outputs) if tensor.is_floating_point()]
    if not tensors:
        return None

    confidence_terms = []
    for tensor in tensors:
        if tensor.ndim >= 3 and tensor.shape[-1] >= 6:
            scores = tensor[..., 4:]
        elif tensor.ndim >= 3 and tensor.shape[1] >= 6:
            scores = tensor[:, 4:, :]
        else:
            scores = tensor
        confidence_terms.append(torch.sigmoid(scores).mean())

    return torch.stack(confidence_terms).mean()


class NoiseManager:
    def __init__(self):
        self.epsilon = float(os.getenv("NOISE_EPSILON", "10.0"))
        self.pgd_steps = int(os.getenv("NOISE_PGD_STEPS", "8"))
        self.pgd_alpha = float(os.getenv("NOISE_PGD_ALPHA", "1.5"))
        self.surrogate_imgsz = int(os.getenv("NOISE_SURROGATE_IMGSZ", "640"))

        self.grid_strength = float(os.getenv("NOISE_GRID_STRENGTH", "5.0"))
        self.color_shift = float(os.getenv("NOISE_COLOR_SHIFT", "3.0"))
        self.jpeg_quality = int(os.getenv("NOISE_JPEG_QUALITY", "88"))

    def _pgd_perturbation(self, img_array: np.ndarray) -> np.ndarray:
        """Generate PGD perturbation using YOLO26 as the white-box surrogate."""
        model, kind = _get_surrogate_model()
        if model is None:
            return np.zeros_like(img_array, dtype=np.float32)

        try:
            img_tensor = torch.from_numpy(
                img_array.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            attack_tensor = F.interpolate(
                img_tensor,
                size=(self.surrogate_imgsz, self.surrogate_imgsz),
                mode="bilinear",
                align_corners=False,
            )
            delta = torch.empty_like(attack_tensor).uniform_(
                -self.epsilon / 255.0,
                self.epsilon / 255.0,
            )
            delta.requires_grad_(True)

            for _step in range(self.pgd_steps):
                perturbed = torch.clamp(attack_tensor + delta, 0, 1)
                outputs = model(perturbed)

                if kind == "yolo26":
                    confidence = _yolo_confidence(outputs)
                    if confidence is None:
                        break
                    loss = -confidence
                else:
                    logits = outputs
                    pred_class = logits.argmax(dim=1)
                    loss = F.cross_entropy(logits, pred_class)

                loss.backward()
                if delta.grad is None:
                    break

                delta.data = delta.data + (self.pgd_alpha / 255.0) * delta.grad.sign()
                delta.data = torch.clamp(
                    delta.data,
                    -self.epsilon / 255.0,
                    self.epsilon / 255.0,
                )
                delta.data = torch.clamp(attack_tensor + delta.data, 0, 1) - attack_tensor
                delta.grad.zero_()

            perturbation = F.interpolate(
                delta.detach(),
                size=img_array.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            return (
                perturbation.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255.0
            )
        except Exception as exc:
            logger.warning("PGD perturbation failed: %s", exc)
            return np.zeros_like(img_array, dtype=np.float32)

    def _high_freq_grid(self, shape: tuple[int, int, int]) -> np.ndarray:
        h, w, c = shape
        grid = np.zeros(shape, dtype=np.float32)

        y_coords = np.arange(h)
        x_coords = np.arange(w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        for freq in (2, 4, 8):
            pattern = ((yy // freq + xx // freq) % 2) * 2 - 1
            for ch in range(c):
                grid[:, :, ch] += pattern * (self.grid_strength / 3)

        return grid

    def _color_perturbation(self, img_array: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img_array.astype(np.uint8))
        hsv = img.convert("HSV")
        hsv_array = np.array(hsv, dtype=np.float32)

        h_shift = np.random.uniform(-self.color_shift, self.color_shift)
        s_shift = np.random.uniform(-self.color_shift * 2, self.color_shift * 2)

        hsv_array[:, :, 0] = (hsv_array[:, :, 0] + h_shift) % 256
        hsv_array[:, :, 1] = np.clip(hsv_array[:, :, 1] + s_shift, 0, 255)

        perturbed_hsv = Image.fromarray(hsv_array.astype(np.uint8), mode="HSV")
        perturbed_rgb = np.array(perturbed_hsv.convert("RGB"), dtype=np.float32)
        return perturbed_rgb - img_array.astype(np.float32)

    def noise(self, image: bytes) -> str:
        """Performs adversarial noising on an image."""
        img = Image.open(io.BytesIO(image)).convert("RGB")
        img_array = np.array(img, dtype=np.float32)

        try:
            total_perturbation = np.zeros_like(img_array)
            total_perturbation += self._pgd_perturbation(img_array) * 0.70
            total_perturbation += self._high_freq_grid(img_array.shape) * 0.20
            total_perturbation += self._color_perturbation(img_array) * 0.10
            total_perturbation = np.clip(
                total_perturbation,
                -self.epsilon,
                self.epsilon,
            )

            noised = np.clip(img_array + total_perturbation, 0, 255)
            noised_img = Image.fromarray(noised.astype(np.uint8))

            buffered = io.BytesIO()
            noised_img.save(buffered, format="JPEG", quality=self.jpeg_quality)
            return base64.b64encode(buffered.getvalue()).decode("ascii")
        except Exception as exc:
            logger.error("Noise generation failed: %s", exc)
            return base64.b64encode(image).decode("ascii")
