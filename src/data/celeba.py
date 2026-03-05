"""
CelebA Dataset Loading and Preprocessing

This module handles loading and preprocessing the CelebA dataset for
training diffusion models. It includes:
- Loading from HuggingFace Hub (electronickale/cmu-10799-celeba64-subset)
- Loading from local directory (downloaded datasets)

What you need to implement:
- Data preprocessing and postprocessing transform functions
- Data augmentations if needed
"""

import os
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid as torch_make_grid
from torchvision.utils import save_image as torch_save_image
from PIL import Image


class CelebADataset(Dataset):
    """
    CelebA dataset wrapper with preprocessing for diffusion models.

    Supports two modes:
    1. HuggingFace mode: Loads from HuggingFace Hub (electronickale/cmu-10799-celeba64-subset)
    2. Local mode: Loads from local directory with images/ and attributes.csv

    Args:
        root: Root directory for the dataset (e.g., "./data/celeba-subset")
        split: Dataset split ('train', 'validation', or 'all') (currently only 'train' is available)
        image_size: Target image resolution (default: 64, images are already 64x64)
        augment: Whether to apply data augmentation
        from_hub: Whether to load from HuggingFace Hub (default: False, loads locally)
        repo_name: HuggingFace repo name (default: "electronickale/cmu-10799-celeba64-subset")
    """

    def __init__(
        self,
        root: str = "./data/celeba-subset",
        split: str = "train",
        image_size: int = 64,
        augment: bool = True,
        from_hub: bool = False,
        repo_name: str = "electronickale/cmu-10799-celeba64-subset",
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.from_hub = from_hub
        self.repo_name = repo_name

        # Build transforms
        self.transform = self._build_transforms() # TODO write your own image transform function

        # Load dataset based on mode
        if from_hub:
            self._load_from_hub()
        else:
            self._load_from_local()

    def _load_from_hub(self):
        """Load dataset from HuggingFace Hub or cached Arrow format."""
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError:
            raise ImportError(
                "Please install the datasets library to load from HuggingFace Hub:\n"
                "  pip install datasets"
            )

        # First, try to load from local cached dataset if root path is provided
        from pathlib import Path
        root_path = Path(self.root)
        print(f"Attempt to use cached dataset from: {self.root}")
        if root_path.exists() and (root_path / "dataset_dict.json").exists():
            print("=" * 60)
            print(f"✓ Using cached dataset from: {self.root}")
            print("  (No download required - using local Arrow format cache)")
            print("=" * 60)

            # Map split names (HF uses 'validation' not 'valid')
            hf_split = "validation" if self.split == "valid" else self.split

            # Load the dataset from disk
            dataset = load_from_disk(self.root)

            if hf_split == "all":
                # Combine all splits
                all_data = []
                for split_name in dataset.keys():
                    all_data.extend(list(dataset[split_name]))
                self.data = all_data
            else:
                self.data = list(dataset[hf_split])

            print(f"✓ Loaded {len(self.data)} images from cached '{hf_split}' split")
            return

        # Otherwise, download from HuggingFace Hub
        print("=" * 60)
        print(f"⬇ Downloading dataset from HuggingFace Hub: {self.repo_name}")
        print(f"  (This may take a few minutes on first run)")
        print("=" * 60)

        # Map split names (HF uses 'validation' not 'valid')
        hf_split = "validation" if self.split == "valid" else self.split

        cache_dir = None
        if self.root:
            os.makedirs(self.root, exist_ok=True)
            cache_dir = self.root
            print(f"Using HuggingFace cache directory: {self.root}")

        if hf_split == "all":
            self.dataset = load_dataset(self.repo_name, cache_dir=cache_dir)
            # Combine all splits
            all_data = []
            for split_name in self.dataset.keys():
                all_data.extend(list(self.dataset[split_name]))
            self.data = all_data
        else:
            self.dataset = load_dataset(self.repo_name, split=hf_split, cache_dir=cache_dir)
            self.data = list(self.dataset)

        print(f"Loaded {len(self.data)} images from {hf_split} split")

    def _load_from_local(self):
        """Load dataset from local directory."""
        from pathlib import Path

        # First, try loading from HuggingFace saved dataset (Arrow format)
        # This is used when dataset was downloaded with save_to_disk()
        if self._try_load_from_saved_dataset():
            return

        # Otherwise, fall back to loading from image files
        # Map split names for directory structure
        split_dir = self.split
        if self.split == "valid":
            split_dir = "validation"

        # Determine the split directory
        if self.split == "all":
            # Load both train and validation
            train_path = Path(self.root) / "train"
            val_path = Path(self.root) / "validation"

            self.data = []
            if train_path.exists():
                self.data.extend(self._load_split_data(train_path))
            if val_path.exists():
                self.data.extend(self._load_split_data(val_path))
        else:
            split_path = Path(self.root) / split_dir
            self.data = self._load_split_data(split_path)

        print(f"Loaded {len(self.data)} images from local directory")

    def _try_load_from_saved_dataset(self):
        """Try to load from HuggingFace saved dataset format (Arrow).

        Returns True if successful, False otherwise.
        """
        from pathlib import Path

        # Check if this looks like a HuggingFace saved dataset
        root_path = Path(self.root)
        if not root_path.exists():
            return False

        # HuggingFace datasets saved with save_to_disk() have dataset_info.json
        if not (root_path / "dataset_info.json").exists():
            return False

        try:
            from datasets import load_from_disk
        except ImportError:
            return False

        print(f"Loading dataset from saved HuggingFace format: {self.root}")

        # Map split names
        hf_split = "validation" if self.split == "valid" else self.split

        # Load the dataset
        dataset = load_from_disk(self.root)

        if hf_split == "all":
            # Combine all splits
            all_data = []
            for split_name in dataset.keys():
                all_data.extend(list(dataset[split_name]))
            self.data = all_data
        else:
            self.data = list(dataset[hf_split])

        print(f"Loaded {len(self.data)} images from {hf_split} split")
        return True

    def _load_split_data(self, split_path):
        """Load data from a split directory."""
        from pathlib import Path

        images_dir = split_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                f"Please download the dataset first using:\n"
                f"  python dataset_processing/download_dataset.py"
            )

        # Get all image files
        image_files = sorted(images_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(images_dir.glob("*.jpg"))

        # Create data entries
        data = []
        for img_path in image_files:
            data.append({
                "image": str(img_path),
                "image_id": img_path.name,
            })

        return data
    
    def _build_transforms(self) -> Callable:
        """Build the preprocessing transforms."""
        transform_list = []

        # Ensure consistent size (HF images are already 64x64, but keep it safe)
        if self.image_size:
            transform_list.append(transforms.Resize(self.image_size))
            transform_list.append(transforms.CenterCrop(self.image_size))

        # Optional augmentation for training
        if self.augment and self.split == "train":
            transform_list.append(transforms.RandomHorizontalFlip())

        # Convert to tensor and normalize to [-1, 1]
        transform_list.append(transforms.ToTensor())
        transform_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single image.

        Args:
            idx: Index of the image

        Returns:
            Image tensor of shape (3, image_size, image_size) in range [-1, 1]

        Note:
            We only return the image, not the attributes, since we're doing
            unconditional generation.
        """
        item = self.data[idx]

        # Load image
        if self.from_hub:
            # HuggingFace datasets provide PIL images directly
            image = item["image"]
        else:
            # Local mode: load from file path
            image = Image.open(item["image"]).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image


def create_dataloader(
    root: str = "./data/celeba-subset",
    split: str = "train",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
    shuffle: Optional[bool] = None,
    drop_last: bool = True,
    from_hub: bool = False,
    repo_name: str = "electronickale/cmu-10799-celeba64-subset",
) -> DataLoader:
    """
    Create a DataLoader for CelebA.

    Args:
        root: Root directory for local dataset (default: "./data/celeba-subset")
        split: Dataset split ('train', 'validation', or 'all')
        image_size: Target image resolution (default: 64)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle (defaults to True for train, False otherwise)
        drop_last: Whether to drop the last incomplete batch
        from_hub: Whether to load from HuggingFace Hub (default: False)
        repo_name: HuggingFace repo name (default: "electronickale/cmu-10799-celeba64-subset")

    Returns:
        DataLoader instance
    """
    dataset = CelebADataset(
        root=root,
        split=split,
        image_size=image_size,
        augment=augment,
        from_hub=from_hub,
        repo_name=repo_name,
    )

    if shuffle is None:
        shuffle = (split == "train")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


def create_dataloader_from_config(config: dict, split: str = "train") -> DataLoader:
    """
    Create a DataLoader from a configuration dictionary.

    Args:
        config: Configuration dictionary
        split: Dataset split

    Returns:
        DataLoader instance
    """
    data_config = config['data']
    training_config = config['training']

    return create_dataloader(
        root=data_config.get('root', './data/celeba-subset'),
        split=split,
        image_size=data_config['image_size'],
        batch_size=training_config['batch_size'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        augment=(split == "train" and data_config.get('augment', True)),
        from_hub=data_config.get('from_hub', False),
        repo_name=data_config.get('repo_name', 'electronickale/cmu-10799-celeba64-subset'),
    )

"""
Some helper fuctions
"""
def unnormalize(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [-1, 1] to [0, 1] range.

    Args:
        images: Image tensor of shape (B, C, H, W) or (C, H, W) in range [-1, 1]

    Returns:
        Image tensor in range [0, 1]
    """
    return (images + 1.0) / 2.0


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [0, 1] to [-1, 1] range.

    Args:
        images: Image tensor of shape (B, C, H, W) or (C, H, W) in range [0, 1]

    Returns:
        Image tensor in range [-1, 1]
    """
    return images * 2.0 - 1.0


def make_grid(images: torch.Tensor, nrow: int = 8, **kwargs) -> torch.Tensor:
    """
    Create a grid of images.

    Args:
        images: Image tensor of shape (B, C, H, W)
        nrow: Number of images per row
        **kwargs: Additional arguments passed to torchvision.utils.make_grid

    Returns:
        Grid tensor of shape (C, H', W')
    """
    return torch_make_grid(images, nrow=nrow, **kwargs)


def save_image(images: torch.Tensor, path: str, nrow: int = 8, **kwargs):
    """
    Save a batch of images as a grid.

    Args:
        images: Image tensor of shape (B, C, H, W) in range [-1, 1] or [0, 1]
        path: File path to save the image
        nrow: Number of images per row
        **kwargs: Additional arguments passed to torchvision.utils.save_image
    """
    torch_save_image(images, path, nrow=nrow, **kwargs)
