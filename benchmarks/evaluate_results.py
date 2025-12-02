"""
CLIP Score Evaluation for Generated Images

This script calculates CLIP scores for images in the results/ folder
using prompts from prompts.csv.
"""

import torch
import clip
from PIL import Image
from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm


class ResultsEvaluator:
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

    def load_prompts_from_csv(self, csv_path: str) -> dict:
        """
        Load prompts from CSV file.

        Args:
            csv_path: Path to prompts.csv

        Returns:
            Dictionary mapping image index to prompt text
        """
        prompts = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for idx, row in enumerate(reader, 1):
                if row:  # Skip empty rows
                    prompts[idx] = row[0].strip()
        return prompts

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

    def evaluate_results(self, results_dir: str, prompts_csv: str):
        """
        Evaluate all generated images in results directory.

        Args:
            results_dir: Directory containing generated images (e.g., 'results')
            prompts_csv: Path to CSV file with prompts (e.g., 'prompts.csv')

        Returns:
            Tuple of (dict of scores, list of scores, mean score)
        """
        results_path = Path(results_dir)

        # Load prompts
        print(f"Loading prompts from: {prompts_csv}")
        prompts = self.load_prompts_from_csv(prompts_csv)
        print(f"Loaded {len(prompts)} prompts\n")

        # Find all generated images
        image_files = sorted(results_path.glob("image_*.png"))

        if not image_files:
            raise ValueError(f"No images found in {results_dir}")

        print(f"Found {len(image_files)} generated images\n")
        print("Calculating CLIP scores...\n")

        scores_dict = {}
        scores_list = []
        skipped = []

        for image_path in tqdm(image_files, desc="Processing"):
            # Extract index from filename (e.g., image_001.png -> 1)
            filename = image_path.stem
            try:
                idx = int(filename.split('_')[1])
            except (IndexError, ValueError):
                print(f"Warning: Couldn't parse index from {image_path.name}, skipping...")
                continue

            # Get corresponding prompt
            if idx not in prompts:
                print(f"Warning: No prompt found for image {idx}, skipping...")
                skipped.append(idx)
                continue

            prompt = prompts[idx]

            # Calculate score
            score = self.calculate_clip_score(str(image_path), prompt)
            scores_dict[idx] = {
                'filename': image_path.name,
                'prompt': prompt,
                'score': score
            }
            scores_list.append(score)

        mean_score = np.mean(scores_list) if scores_list else 0.0

        if skipped:
            print(f"\nSkipped {len(skipped)} images due to missing prompts: {skipped}")

        return scores_dict, scores_list, mean_score


def main():
    """Main function to evaluate generated images."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CLIP scores for generated images")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing generated images (default: results)"
    )
    parser.add_argument(
        "--prompts_csv",
        type=str,
        default="prompts.csv",
        help="CSV file with prompts (default: prompts.csv)"
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
        default="clip_scores.csv",
        help="Output file to save scores (default: clip_scores.csv)"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ResultsEvaluator(model_name=args.model)

    # Evaluate images
    scores_dict, scores_list, mean_score = evaluator.evaluate_results(
        args.results_dir,
        args.prompts_csv
    )

    # Print results
    print("\n" + "="*60)
    print("CLIP Score Results")
    print("="*60)
    print(f"Total images evaluated: {len(scores_list)}")
    print(f"Mean CLIP Score: {mean_score:.4f}")
    if scores_list:
        print(f"Std CLIP Score: {np.std(scores_list):.4f}")
        print(f"Min CLIP Score: {np.min(scores_list):.4f}")
        print(f"Max CLIP Score: {np.max(scores_list):.4f}")
    print("="*60)

    # Save detailed scores with summary statistics
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Write summary statistics at the top
        writer.writerow(["Summary Statistics"])
        writer.writerow(["Total Images Evaluated", len(scores_list)])
        writer.writerow(["Mean CLIP Score", f"{mean_score:.6f}"])
        if scores_list:
            writer.writerow(["Std CLIP Score", f"{np.std(scores_list):.6f}"])
            writer.writerow(["Min CLIP Score", f"{np.min(scores_list):.6f}"])
            writer.writerow(["Max CLIP Score", f"{np.max(scores_list):.6f}"])
        writer.writerow([])  # Empty row separator

        # Write individual scores
        writer.writerow(["image_id", "filename", "clip_score", "prompt"])
        for idx in sorted(scores_dict.keys()):
            data = scores_dict[idx]
            writer.writerow([
                idx,
                data['filename'],
                f"{data['score']:.6f}",
                data['prompt']
            ])

    print(f"\nDetailed scores saved to: {args.output}")


if __name__ == "__main__":
    main()
