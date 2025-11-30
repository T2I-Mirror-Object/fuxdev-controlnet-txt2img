"""
FID (Fréchet Inception Distance) Score Calculator

This script calculates FID scores to measure the quality and diversity
of generated images compared to real images using Inception-v3 features.

Lower FID scores indicate better image quality and similarity to real data.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
from scipy import linalg
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights


class FIDScoreCalculator:
    def __init__(self, device: str = None, dims: int = 2048):
        """
        Initialize Inception-v3 model for FID calculation.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            dims: Dimensionality of Inception features (default: 2048)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dims = dims

        print(f"Loading Inception-v3 model on {self.device}...")

        # Load pretrained Inception-v3
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()  # Remove final classification layer
        self.model.to(self.device)
        self.model.eval()

        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Inception-v3 model loaded successfully!\n")

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract Inception features from a single image.

        Args:
            image_path: Path to image file

        Returns:
            Feature vector (2048-dimensional)
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.cpu().numpy().flatten()

    def extract_features_from_directory(self, image_dir: str, batch_size: int = 32) -> np.ndarray:
        """
        Extract features from all images in a directory.

        Args:
            image_dir: Directory containing images
            batch_size: Batch size for processing

        Returns:
            Array of features with shape (num_images, 2048)
        """
        data_path = Path(image_dir)
        image_files = sorted(list(data_path.glob("*.png")) + list(data_path.glob("*.jpg")))

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Extracting features from {len(image_files)} images...")

        all_features = []

        # Process in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_files = image_files[i:i + batch_size]
            batch_images = []

            for img_path in batch_files:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(self.transform(image))

            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                batch_features = self.model(batch_tensor)

            all_features.append(batch_features.cpu().numpy())

        return np.vstack(all_features)

    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and covariance of features.

        Args:
            features: Feature array with shape (num_images, dims)

        Returns:
            Tuple of (mean, covariance)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Calculate FID score between two distributions.

        Args:
            mu1: Mean of distribution 1
            sigma1: Covariance of distribution 1
            mu2: Mean of distribution 2
            sigma2: Covariance of distribution 2
            eps: Epsilon for numerical stability

        Returns:
            FID score
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return float(fid)

    def calculate_fid_from_directories(
        self,
        generated_dir: str,
        real_dir: str,
        batch_size: int = 32
    ) -> float:
        """
        Calculate FID score between generated and real images.

        Args:
            generated_dir: Directory containing generated images
            real_dir: Directory containing real/reference images
            batch_size: Batch size for feature extraction

        Returns:
            FID score
        """
        # Extract features from both directories
        print("\n[1/3] Extracting features from generated images...")
        gen_features = self.extract_features_from_directory(generated_dir, batch_size)

        print("\n[2/3] Extracting features from real images...")
        real_features = self.extract_features_from_directory(real_dir, batch_size)

        # Calculate statistics
        print("\n[3/3] Computing FID score...")
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        mu_real, sigma_real = self.calculate_statistics(real_features)

        # Calculate FID
        fid_score = self.calculate_fid(mu_gen, sigma_gen, mu_real, sigma_real)

        return fid_score


def main():
    """Main function to calculate FID score."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate FID score for image generation")
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Directory containing generated images"
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Directory containing real/reference images"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)"
    )

    args = parser.parse_args()

    # Validate directories
    if not Path(args.generated_dir).exists():
        raise ValueError(f"Generated directory not found: {args.generated_dir}")
    if not Path(args.real_dir).exists():
        raise ValueError(f"Real directory not found: {args.real_dir}")

    # Initialize calculator
    calculator = FIDScoreCalculator()

    # Calculate FID
    fid_score = calculator.calculate_fid_from_directories(
        args.generated_dir,
        args.real_dir,
        args.batch_size
    )

    # Print results
    print("\n" + "="*60)
    print("FID Score Results")
    print("="*60)
    print(f"Generated images: {args.generated_dir}")
    print(f"Real images: {args.real_dir}")
    print(f"FID Score: {fid_score:.4f}")
    print("="*60)
    print("\nNote: Lower FID scores indicate better quality and diversity")
    print("      Typical ranges: 0-20 (excellent), 20-50 (good), >50 (poor)")
    print("="*60)


if __name__ == "__main__":
    main()
