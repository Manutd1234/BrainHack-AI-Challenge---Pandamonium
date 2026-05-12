"""Manages the noise model."""

import base64
import io

import numpy as np
from PIL import Image


class NoiseManager:

    def __init__(self):
        # Configuration for adversarial noise.
        # noise_strength controls the intensity of the perturbation.
        # Keep it moderate to stay within SSIM/RMSE thresholds.
        self.noise_strength = 15  # pixel intensity delta (0-255 scale)

    def noise(self, image: bytes) -> str:
        """Performs adversarial noising on an image.

        Adds subtle random noise to the image to disrupt opponent CV models
        while staying within the SSIM and RMSE L2 norm evaluation thresholds.

        Args:
            image: The image file in bytes (JPEG format).

        Returns:
            A string containing your output image encoded in base64.
        """

        img = Image.open(io.BytesIO(image))
        img_array = np.array(img, dtype=np.float32)

        try:
            # Generate random noise
            noise = np.random.uniform(
                -self.noise_strength,
                self.noise_strength,
                img_array.shape,
            ).astype(np.float32)

            # Apply noise and clip to valid range
            noised = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            # Convert back to JPEG bytes and encode as base64
            noised_img = Image.fromarray(noised)
            buffered = io.BytesIO()
            noised_img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("ascii")

        except Exception as e:
            print(f"Error occurred: {e}")
            # On failure, return the original image unchanged
            return base64.b64encode(image).decode("ascii")
