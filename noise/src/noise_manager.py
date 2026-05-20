"""PGD adversarial noising against a surrogate YOLO CV model."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)

SURROGATE_PATH = os.getenv("CV_MODEL_PATH", "/app/model/best.pt")
EPSILON = float(os.getenv("NOISE_EPSILON", str(10 / 255.0)))
ALPHA = float(os.getenv("NOISE_ALPHA", str(3 / 255.0)))
PGD_STEPS = int(os.getenv("NOISE_PGD_STEPS", "7"))
SSIM_MIN = float(os.getenv("NOISE_SSIM_MIN", "0.85"))
BISECT_ITERS = int(os.getenv("NOISE_BISECT_ITERS", "8"))
JPEG_QUALITY = int(os.getenv("NOISE_JPEG_QUALITY", "95"))
MAX_SIDE = int(os.getenv("NOISE_MAX_SIDE", "640"))


class NoiseManager:
    """Creates visually similar adversarial JPEGs for the Noise challenge."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_surrogate()

    def noise(self, image: bytes) -> str:
        """Return a base64-encoded adversarial JPEG for an input JPEG byte string."""
        try:
            original_bgr = self._decode_bytes(image)
            attacked_bgr = self._attack_image(original_bgr)
            return self._encode_b64(attacked_bgr)
        except Exception as exc:
            LOGGER.exception("Noise generation failed; returning original image: %s", exc)
            return base64.b64encode(image).decode("ascii")

    def add_noise(self, b64_jpeg: str) -> str:
        """Compatibility helper for standalone clients."""
        return self.noise(base64.b64decode(b64_jpeg))

    def _load_surrogate(self) -> None:
        if not os.path.exists(SURROGATE_PATH):
            LOGGER.warning(
                "Surrogate CV model not found at %s; using deterministic fallback noise.",
                SURROGATE_PATH,
            )
            return

        LOGGER.info("Loading YOLO surrogate from %s on %s", SURROGATE_PATH, self.device)
        yolo = YOLO(SURROGATE_PATH)
        self.model = yolo.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        LOGGER.info("NoiseManager ready")

    def _decode_bytes(self, image_bytes: bytes) -> np.ndarray:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        return image

    def _encode_b64(self, image_bgr: np.ndarray) -> str:
        ok, buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not ok:
            raise ValueError("Failed to encode adversarial image as JPEG")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def _attack_image(self, original_bgr: np.ndarray) -> np.ndarray:
        attack_bgr = self._resize_for_attack(original_bgr)

        if self.model is None:
            noised_bgr = self._fallback_noise(attack_bgr)
        else:
            original_tensor = self._to_tensor(attack_bgr)
            adversarial_tensor = self._pgd(original_tensor)
            noised_bgr = self._to_numpy(adversarial_tensor)

        projected_bgr = self._project_ssim(attack_bgr, noised_bgr)
        if projected_bgr.shape[:2] != original_bgr.shape[:2]:
            projected_bgr = cv2.resize(
                projected_bgr,
                (original_bgr.shape[1], original_bgr.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            projected_bgr = self._project_ssim(original_bgr, projected_bgr)

        return projected_bgr

    def _resize_for_attack(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        largest = max(height, width)
        if largest <= MAX_SIDE:
            return image_bgr

        scale = MAX_SIDE / largest
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        rgb = (
            tensor.squeeze(0)
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            .clip(0.0, 1.0)
        )
        rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    def _pgd(self, original: torch.Tensor) -> torch.Tensor:
        x = original.detach() + torch.empty_like(original).uniform_(-EPSILON, EPSILON)
        x = x.clamp(0.0, 1.0)

        use_amp = self.device == "cuda"
        for _ in range(PGD_STEPS):
            x = x.detach().requires_grad_(True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss = self._detection_loss(x)
            else:
                loss = self._detection_loss(x)
            loss.backward()

            with torch.no_grad():
                gradient = x.grad.sign()
                x = x + ALPHA * gradient
                delta = (x - original).clamp(-EPSILON, EPSILON)
                x = (original + delta).clamp(0.0, 1.0)

        return x.detach()

    def _detection_loss(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        tensors = self._flatten_tensors(outputs)
        if not tensors:
            return x.mean() * 0.0

        losses = []
        for tensor in tensors:
            if tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                continue
            candidate = tensor.float()
            scores = self._detection_scores(candidate)
            if scores is not None:
                confidence = scores.sigmoid()
                losses.append(confidence.mean() + 0.50 * confidence.amax())
            else:
                losses.append(candidate.square().mean())

        if not losses:
            return x.mean() * 0.0
        return torch.stack([loss.reshape(()) for loss in losses]).mean()

    def _detection_scores(self, tensor: torch.Tensor) -> torch.Tensor | None:
        if tensor.ndim < 3:
            return None

        if 5 <= tensor.shape[1] <= 256 and tensor.shape[-1] > tensor.shape[1]:
            channels_first = tensor
            return channels_first[:, 4:, :]

        if 5 <= tensor.shape[-1] <= 256:
            channels_last = tensor
            return channels_last[..., 4:]

        return None

    def _flatten_tensors(self, value: Any) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            tensors = []
            for item in value.values():
                tensors.extend(self._flatten_tensors(item))
            return tensors
        if isinstance(value, (list, tuple)):
            tensors = []
            for item in value:
                tensors.extend(self._flatten_tensors(item))
            return tensors
        return []

    def _fallback_noise(self, original_bgr: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(2026)
        noise = rng.normal(0.0, EPSILON * 255.0 / 2.0, original_bgr.shape)
        checker = np.indices(original_bgr.shape[:2]).sum(axis=0) % 2
        checker = (checker[..., None] * 2 - 1) * EPSILON * 255.0 * 0.35
        noised = original_bgr.astype(np.float32) + noise + checker
        return np.clip(noised, 0, 255).astype(np.uint8)

    def _project_ssim(self, original_bgr: np.ndarray, noised_bgr: np.ndarray) -> np.ndarray:
        current_ssim = self._ssim(original_bgr, noised_bgr)
        if current_ssim >= SSIM_MIN:
            return noised_bgr

        delta = noised_bgr.astype(np.float32) - original_bgr.astype(np.float32)
        low = 0.0
        high = 1.0

        for _ in range(BISECT_ITERS):
            mid = (low + high) / 2.0
            candidate = np.clip(
                original_bgr.astype(np.float32) + delta * mid,
                0,
                255,
            ).astype(np.uint8)
            if self._ssim(original_bgr, candidate) >= SSIM_MIN:
                low = mid
            else:
                high = mid

        return np.clip(
            original_bgr.astype(np.float32) + delta * low,
            0,
            255,
        ).astype(np.uint8)

    def _ssim(self, original_bgr: np.ndarray, candidate_bgr: np.ndarray) -> float:
        min_side = min(original_bgr.shape[:2])
        if min_side < 3:
            mse = np.mean(
                (
                    original_bgr.astype(np.float32)
                    - candidate_bgr.astype(np.float32)
                )
                ** 2
            )
            return float(max(0.0, 1.0 - mse / (255.0**2)))

        win_size = min(3, min_side if min_side % 2 == 1 else min_side - 1)
        return float(
            ssim_metric(
                original_bgr,
                candidate_bgr,
                channel_axis=2,
                data_range=255,
                win_size=win_size,
            )
        )
