# Image Generation Benchmarks

This folder contains scripts to evaluate text-to-image generation quality using two key metrics:

## Metrics

### 1. CLIP Score (`clip_score.py`)
Measures the **semantic alignment** between generated images and text prompts using OpenAI's CLIP model.

- **Higher scores = better** text-image alignment
- Range: typically 0.0 to 1.0 (cosine similarity)
- Evaluates how well the image matches the text description

### 2. FID Score (`fid_score.py`)
Measures the **quality and diversity** of generated images compared to real images using Inception-v3 features.

- **Lower scores = better** quality
- Range: 0 to ∞
  - 0-20: Excellent quality
  - 20-50: Good quality
  - >50: Poor quality

## Installation

Install required dependencies:

```bash
pip install torch torchvision clip-by-openai pillow numpy scipy tqdm
```

Or install CLIP from GitHub:
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Usage

### CLIP Score

Calculate CLIP scores for image-text pairs in the `data/` folder:

```bash
cd benchmarks
python clip_score.py --data_dir data
```

Options:
- `--data_dir`: Directory with image-text pairs (default: `data`)
- `--model`: CLIP model variant (default: `ViT-B/32`)
  - Options: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, etc.
- `--output`: Save individual scores to CSV file

Example with custom output:
```bash
python clip_score.py --data_dir data --model ViT-L/14 --output clip_scores.csv
```

### FID Score

Calculate FID score between generated and real images:

```bash
cd benchmarks
python fid_score.py --generated_dir data --real_dir /path/to/real/images
```

Options:
- `--generated_dir`: Directory with generated images (required)
- `--real_dir`: Directory with real/reference images (required)
- `--batch_size`: Batch size for processing (default: 32)

Example:
```bash
python fid_score.py --generated_dir ../results --real_dir ./real_images --batch_size 16
```

## Data Format

Both scripts expect images in standard formats (PNG, JPG).

For CLIP score, the directory should contain:
```
data/
  1.png
  1.txt
  2.png
  2.txt
  ...
```

Each `.txt` file contains the text prompt for the corresponding image.

## Output

### CLIP Score Output
```
CLIP Score Results
============================================================
Total images evaluated: 50
Mean CLIP Score: 0.3145
Std CLIP Score: 0.0234
Min CLIP Score: 0.2567
Max CLIP Score: 0.3789
============================================================
```

### FID Score Output
```
FID Score Results
============================================================
Generated images: data
Real images: real_images
FID Score: 25.4321
============================================================
```
