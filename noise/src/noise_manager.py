"""Manages adversarial noise generation.

Uses a combination of techniques to disrupt opponent CV models while
staying within SSIM and RMSE L2 norm evaluation thresholds:

1. PGD (Projected Gradient Descent): Multi-step iterative adversarial attack
   using a surrogate model. Stronger than single-step FGSM and transfers
   better across architectures.
2. High-frequency grid patterns: Disrupt feature extraction in early
   conv layers.
3. Color channel perturbation: Subtle HSV shifts that confuse detectors.
4. JPEG re-compression: Introduces artifacts that degrade detection.
"""

import base64
import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import torch for surrogate-model attacks
_torch_available = False
_surrogate_model = None
try:
    import torch
    import torch.nn.functional as F
    _torch_available = True
except ImportError:
    pass


def _get_surrogate_model():
    """Lazy-load a lightweight surrogate model for PGD attacks."""
    global _surrogate_model
    if _surrogate_model is None and _torch_available:
        try:
            import torchvision.models as models
            _surrogate_model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
            _surrogate_model.eval()
            if torch.cuda.is_available():
                _surrogate_model = _surrogate_model.cuda()
            logger.info("Loaded MobileNetV2 surrogate model for PGD.")
        except Exception as e:
            logger.warning(f"Failed to load surrogate model: {e}")
    return _surrogate_model


class NoiseManager:

    def __init__(self):
        # PGD attack parameters
        self.epsilon = 12.0       # total perturbation budget (L∞, 0-255 scale)
        self.pgd_steps = 10       # number of PGD iterations
        self.pgd_alpha = 2.0      # step size per iteration

        # Supplementary noise parameters
        self.grid_strength = 6.0  # high-frequency grid amplitude
        self.color_shift = 4.0    # HSV color perturbation magnitude
        self.jpeg_quality = 85    # JPEG re-compression quality

    def _pgd_perturbation(self, img_array: np.ndarray) -> np.ndarray:
        """Generate PGD adversarial perturbation using a surrogate model.

        Multi-step iterative attack — much stronger than single-step FGSM.
        Perturbations from PGD transfer well across model architectures.
        """
        model = _get_surrogate_model()
        if model is None:
            return np.zeros_like(img_array, dtype=np.float32)

        try:
            # Normalize to [0, 1] for the model
            img_tensor = torch.from_numpy(
                img_array.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)

            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            # Initialize perturbation with small random noise
            delta = torch.zeros_like(img_tensor).uniform_(
                -self.epsilon / 255.0, self.epsilon / 255.0
            )
            delta.requires_grad_(True)

            for step in range(self.pgd_steps):
                perturbed = torch.clamp(img_tensor + delta, 0, 1)

                output = model(perturbed)

                # Untargeted attack: maximize loss on predicted class
                pred_class = output.argmax(dim=1)
                loss = F.cross_entropy(output, pred_class)
                loss.backward()

                # PGD step: move in gradient direction, project to L∞ ball
                grad = delta.grad.data
                delta.data = delta.data + (self.pgd_alpha / 255.0) * grad.sign()
                delta.data = torch.clamp(
                    delta.data,
                    -self.epsilon / 255.0,
                    self.epsilon / 255.0,
                )
                delta.data = torch.clamp(
                    img_tensor + delta.data, 0, 1
                ) - img_tensor
                delta.grad.zero_()

            # Convert back to pixel space
            perturbation = (
                delta.detach()
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255.0
            )

            return perturbation

        except Exception as e:
            logger.warning(f"PGD perturbation failed: {e}")
            return np.zeros_like(img_array, dtype=np.float32)

    def _high_freq_grid(self, shape: tuple) -> np.ndarray:
        """Generate high-frequency grid patterns that disrupt early conv layers."""
        h, w, c = shape
        grid = np.zeros(shape, dtype=np.float32)

        for freq in [2, 4, 8]:
            y_coords = np.arange(h)
            x_coords = np.arange(w)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
            pattern = ((yy // freq + xx // freq) % 2) * 2 - 1
            for ch in range(c):
                grid[:, :, ch] += pattern * (self.grid_strength / 3)

        return grid

    def _color_perturbation(self, img_array: np.ndarray) -> np.ndarray:
        """Apply subtle color channel perturbations in HSV space."""
        img = Image.fromarray(img_array.astype(np.uint8))
        hsv = img.convert("HSV")
        hsv_array = np.array(hsv, dtype=np.float32)

        h_shift = np.random.uniform(-self.color_shift, self.color_shift)
        s_shift = np.random.uniform(
            -self.color_shift * 2, self.color_shift * 2
        )

        hsv_array[:, :, 0] = (hsv_array[:, :, 0] + h_shift) % 256
        hsv_array[:, :, 1] = np.clip(
            hsv_array[:, :, 1] + s_shift, 0, 255
        )

        perturbed_hsv = Image.fromarray(hsv_array.astype(np.uint8), mode="HSV")
        perturbed_rgb = np.array(perturbed_hsv.convert("RGB"), dtype=np.float32)

        return perturbed_rgb - img_array.astype(np.float32)

    def noise(self, image: bytes) -> str:
        """Performs adversarial noising on an image.

        Combines PGD attack with supplementary perturbation techniques
        to maximally disrupt opponent CV models while staying within
        SSIM and RMSE thresholds.

        Args:
            image: The image file in bytes (JPEG format).

        Returns:
            A string containing your output image encoded in base64.
        """

        img = Image.open(io.BytesIO(image))
        img_array = np.array(img, dtype=np.float32)

        try:
            total_perturbation = np.zeros_like(img_array)

            # 1. PGD adversarial perturbation (primary attack, strongest signal)
            pgd_noise = self._pgd_perturbation(img_array)
            total_perturbation += pgd_noise * 0.6

            # 2. High-frequency grid patterns (supplementary)
            grid_noise = self._high_freq_grid(img_array.shape)
            total_perturbation += grid_noise * 0.25

            # 3. Color channel perturbation (supplementary)
            color_noise = self._color_perturbation(img_array)
            total_perturbation += color_noise * 0.15

            # Clip total perturbation to L∞ budget
            total_perturbation = np.clip(
                total_perturbation, -self.epsilon, self.epsilon
            )

            # Apply perturbation
            noised = np.clip(img_array + total_perturbation, 0, 255)
            noised_img = Image.fromarray(noised.astype(np.uint8))

            # 4. JPEG re-compression (additional artifact noise)
            buffered = io.BytesIO()
            noised_img.save(
                buffered, format="JPEG", quality=self.jpeg_quality
            )
            return base64.b64encode(buffered.getvalue()).decode("ascii")

        except Exception as e:
            logger.error(f"Noise generation failed: {e}")
            return base64.b64encode(image).decode("ascii")
