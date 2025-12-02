"""
CLIP Score Calculator for Text-to-Image Generation

This script calculates CLIP scores to measure the alignment between
generated images and their text prompts using OpenAI's CLIP model.

Higher CLIP scores indicate better text-image alignment.
"""

import torch
import clip
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


class CLIPScoreCalculator:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP model for score calculation.

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print("CLIP model loaded successfully!\n")

    def calculate_clip_score(self, image_path: str, text: str) -> float:
        """
        Calculate CLIP score for a single image-text pair.

        Args:
            image_path: Path to the image file
            text: Text prompt

        Returns:
            CLIP score (cosine similarity between image and text embeddings)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize text
        text_input = clip.tokenize([text], truncate=True).to(self.device)

        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            clip_score = (image_features @ text_features.T).item()

        return clip_score

    def calculate_dataset_scores(self, data_dir: str) -> Tuple[List[float], float]:
        """
        Calculate CLIP scores for all image-text pairs in a directory.

        Expects directory structure:
            data_dir/
                1.png, 1.txt
                2.png, 2.txt
                ...

        Args:
            data_dir: Directory containing image-text pairs

        Returns:
            Tuple of (list of individual scores, mean score)
        """
        data_path = Path(data_dir)

        # Find all image files
        image_files = sorted(data_path.glob("*.png")) + sorted(data_path.glob("*.jpg"))

        if not image_files:
            raise ValueError(f"No images found in {data_dir}")

        scores = []
        print(f"Calculating CLIP scores for {len(image_files)} images...\n")

        for image_path in tqdm(image_files, desc="Processing"):
            # Get corresponding text file
            text_path = image_path.with_suffix('.txt')

            if not text_path.exists():
                print(f"Warning: No text file found for {image_path.name}, skipping...")
                continue

            # Read text prompt
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # Calculate score
            score = self.calculate_clip_score(str(image_path), text)
            scores.append(score)

        mean_score = np.mean(scores)
        return scores, mean_score


def main():
    """Main function to calculate CLIP scores."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate CLIP scores for text-to-image generation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing image-text pairs (default: data)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model variant (default: ViT-B/32)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save individual scores (optional)"
    )

    args = parser.parse_args()

    # Initialize calculator
    calculator = CLIPScoreCalculator(model_name=args.model)

    # Calculate scores
    scores, mean_score = calculator.calculate_dataset_scores(args.data_dir)

    # Print results
    print("\n" + "="*60)
    print("CLIP Score Results")
    print("="*60)
    print(f"Total images evaluated: {len(scores)}")
    print(f"Mean CLIP Score: {mean_score:.4f}")
    print(f"Std CLIP Score: {np.std(scores):.4f}")
    print(f"Min CLIP Score: {np.min(scores):.4f}")
    print(f"Max CLIP Score: {np.max(scores):.4f}")
    print("="*60)

    # Save individual scores if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write("image_id,clip_score\n")
            for i, score in enumerate(scores, 1):
                f.write(f"{i},{score:.6f}\n")
        print(f"\nIndividual scores saved to: {args.output}")


if __name__ == "__main__":
    main()
