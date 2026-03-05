"""
Download CelebA Subset from Hugging Face

This script downloads the course dataset from Hugging Face Hub.

Usage:
    python download_dataset.py --output_dir ./data/celeba-subset
    
    # Or specify a custom dataset
    python download_dataset.py --repo electronickale/cmu-10799-celeba64-subset --output_dir ./data
"""

import os
import argparse
from pathlib import Path


def download_from_huggingface(
    repo_name: str = "electronickale/cmu-10799-celeba64-subset",
    output_dir: str = "./data",
    split: str = "train",
):
    """
    Download dataset from Hugging Face Hub.
    
    Args:
        repo_name: HuggingFace repo name
        output_dir: Output directory
        split: Which split to download ('train', 'validation', or 'all')
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return
    
    print("=" * 60)
    print("Downloading CelebA Subset")
    print("=" * 60)
    print(f"Repository: {repo_name}")
    print(f"Output: {output_dir}")
    print(f"Split: {split}")
    print()
    
    # Load dataset
    print("Downloading from Hugging Face Hub...")
    if split == "all":
        dataset = load_dataset(repo_name)
    else:
        dataset = load_dataset(repo_name, split=split)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to disk in our format
    if split == "all" or hasattr(dataset, 'keys'):
        # Multiple splits
        for split_name in dataset.keys():
            save_split(dataset[split_name], output_dir / split_name)
    else:
        # Single split
        save_split(dataset, output_dir / split)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nDataset saved to: {output_dir}")
    print("\nTo use in training:")
    print(f"  python train.py --method ddpm --config configs/ddpm.yaml")
    print(f"\n  (Make sure data.root in config points to {output_dir})")


def save_split(dataset, output_dir: Path):
    """Save a dataset split to disk."""
    from PIL import Image as PILImage
    import pandas as pd
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {len(dataset)} images to {output_dir}...")
    
    # Extract attribute columns (everything except 'image' and 'image_id')
    all_columns = dataset.column_names
    attr_columns = [c for c in all_columns if c not in ['image', 'image_id']]
    
    # Save images and collect attributes
    attributes = []
    for i, item in enumerate(tqdm(dataset, desc=f"Saving {output_dir.name}")):
        # Save image
        img = item['image']
        img_id = item.get('image_id', f"{i:06d}.jpg")
        img_name = img_id.replace('.jpg', '.png')
        img.save(images_dir / img_name)
        
        # Collect attributes
        attrs = {'image_id': img_id}
        for col in attr_columns:
            attrs[col] = item[col]
        attributes.append(attrs)
    
    # Save attributes
    if attr_columns:
        df = pd.DataFrame(attributes)
        df = df.set_index('image_id')
        df.to_csv(output_dir / "attributes.csv")
        print(f"Saved attributes: {attr_columns[:5]}{'...' if len(attr_columns) > 5 else ''}")


def main():
    parser = argparse.ArgumentParser(description='Download CelebA subset from Hugging Face')
    parser.add_argument('--repo', type=str, default='electronickale/cmu-10799-celeba64-subset',
                        help='HuggingFace repo name')
    parser.add_argument('--output_dir', type=str, default='./data/celeba-subset',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'all'],
                        help='Which split to download')
    
    args = parser.parse_args()
    
    download_from_huggingface(
        repo_name=args.repo,
        output_dir=args.output_dir,
        split=args.split,
    )


if __name__ == '__main__':
    main()
