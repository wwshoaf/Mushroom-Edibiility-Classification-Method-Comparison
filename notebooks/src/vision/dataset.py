# dataset.py — PyTorch Dataset for DF20 mushroom images
# Filters DF20 to the 45 species that overlap with the tabular dataset
# applies binary edibility labels from edibility_map.py, and builds train/val/test DataLoaders for ResNet18 fine-tuning.
#
# Exports:
#   DF20EdibilityDataset — PyTorch Dataset subclass
#   build_dataloaders — helper function to build all three splits

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import csv
from collections import defaultdict
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.vision.edibility_map import EDIBILITY_MAP, get_label

# ImageNet normalisation constants
# pretrained on ImageNet
# The model's weights were learned with inputs normalised to these mean/std values
# apply the same normalisation or the pretrained features won't transfer.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmented transform applied only during training.
# Augmentation artificially increases dataset variety
# helps generalise to mushroom photos taken from different sources
# Resize to img_size+32 then RandomCrop to img_size
# forces the model to learn from different spatial positions
# RandomHorizontalFlip/RandomVerticalFlip: more variety
# ColorJitter: simulates different lighting conditions and camera settings.
# RandomRotation(15): ±15 degrees handles slightly tilted photos.
# ToTensor: converts PIL Image [0,255] to float tensor [0.0, 1.0]
# Normalize: applies ImageNet mean/std to match pretrained ResNet18 inputs
def get_train_transform(img_size: int = 224) -> transforms.Compose:
    
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# Deterministic transform applied to val and test sets.
# No augmentation for reproducible predictions.
# resize to the target size, convert to tensor, and normalise.
def get_val_transform(img_size: int = 224) -> transforms.Compose:
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# PyTorch Dataset subclass for DF20 mushroom images
# Inheriting from Dataset requires implementing __len__ and __getitem__,
# The dataset is filtered to the 45 species that:
#   Appear in both DF20 and the Secondary Mushroom Dataset
#   Have a clear edibility classification in edibility_map.py
#   Have at least 50 images
# Each sample is a tuple: (image_tensor, edibility_label, species_name)
# The species_name is passed through so evaluate_cnn.py can compute per-species accuracy for the bridge analysis.
class DF20EdibilityDataset(Dataset):
    
    def __init__(
        self,
        # path to DF20-train_metadata_PROD-2.csv
        metadata_csv: str,
        # path to extracted DF20_300 folder
        image_root: str,
        transform: Optional[transforms.Compose] = None,
        # 'train', 'val', or 'test'
        split: str = 'train',
        val_frac: float = 0.15,
        test_frac: float = 0.15, 
        random_seed: int = 26,
        min_images: int = 50,
    ):
        assert split in ('train', 'val', 'test'), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.image_root = Path(image_root)
        self.transform = transform
        self.split = split

        # Read the metadata CSV and collect all rows whose species appears in EDIBILITY_MAP
        # csv.DictReader reads each row as a dict keyed by column name
        # store each sample as (image_path, label, species_name)
        usable_species = {s for s in EDIBILITY_MAP}
        samples_by_species = defaultdict(list)

        with open(metadata_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                species = row['species'].strip()
                img_path = row['image_path'].strip()
                if species not in usable_species:
                    continue
                label = get_label(species)
                # unknown edibility — skip
                if label == -1:
                    continue
                samples_by_species[species].append((img_path, label, species))

        # Drop species that don't meet the minimum image threshold.
        # Species with very few images produce unreliable per-species accuracy
        # estimates — e.g. Hebeloma crustuliniforme had only 8 test images,
        # making its 37.5% accuracy statistically meaningless.
        all_samples = []
        self.species_list = []
        for species, samples in sorted(samples_by_species.items()):
            if len(samples) >= min_images:
                all_samples.extend(samples)
                self.species_list.append(species)

        print(f"Loaded {len(all_samples):,} images across "
              f"{len(self.species_list)} species")

        # Stratified split per species
        # rng.permutation shuffles indices randomly but reproducibly.
        rng = np.random.default_rng(random_seed)
        train_s, val_s, test_s = [], [], []

        for species in self.species_list:
            sp_samples = [s for s in all_samples if s[2] == species]
            n = len(sp_samples)
            idx = rng.permutation(n)

            # max(1, ...) ensures at least one sample per species per split
            n_test = max(1, int(n * test_frac))
            n_val = max(1, int(n * val_frac))
            n_train = n - n_test - n_val

            train_s.extend([sp_samples[i] for i in idx[:n_train]])
            val_s.extend([sp_samples[i] for i in idx[n_train:n_train+n_val]])
            test_s.extend([sp_samples[i] for i in idx[n_train+n_val:]])

        # Select the correct split
        split_map = {'train': train_s, 'val': val_s, 'test': test_s}
        self.samples = split_map[split]

        print(f"Split '{split}': {len(self.samples):,} samples")

    def __len__(self) -> int:
        # tells DataLoader how many samples are in the dataset
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Called by DataLoader for each sample in a batch.
        # Returns (image_tensor, label_tensor, species_name).
        img_path, label, species = self.samples[idx]

        # DF20 image_path column is a filename (e.g. 2237851949-74654.jpg) with no subdirectory so construct full path by joining with image_root.
        full_path = self.image_root / img_path
        if not full_path.exists():
            # try just the basename in case path format varies
            full_path = self.image_root / Path(img_path).name

        try:
            # convert('RGB') ensures 3-channel output even for grayscale/RGBA
            img = Image.open(full_path).convert('RGB')
        except Exception:
            # If the image file is corrupted or missing, return a neutral grey
            # image rather than crashing the entire training run.
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        # label must be float32 for BCELoss because int labels cause a type error
        return img, torch.tensor(label, dtype=torch.float32), species

    # Compute per-sample weights for WeightedRandomSampler
    # within individual species the ratio varies
    # Weight formula: n_total / (2 * n_class)
    # This gives each class equal total weight regardless of its count.
    def get_class_weights(self) -> torch.Tensor:
        
        labels = [s[1] for s in self.samples]
        n_total = len(labels)
        # not safe count
        n_pos = sum(labels)
        # edible count
        n_neg = n_total - n_pos   
        weight_pos = n_total / (2 * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_total / (2 * n_neg) if n_neg > 0 else 1.0
        weights = [weight_pos if l == 1 else weight_neg for l in labels]
        return torch.tensor(weights, dtype=torch.float32)

# Convenience function that creates all three dataset splits and wraps them in DataLoaders
# Called by train_cnn.py and evaluate_cnn.py.
#
# num_workers=0 is for my pc
#
# WeightedRandomSampler on the training loader ensures balanced batches.
# Val and test loaders use shuffle=False for reproducible evaluation.
#
# Returns:
# train_loader, val_loader, test_loader : DataLoader
def build_dataloaders(
    metadata_csv: str,
    image_root: str,
    # ResNet18 default input 
    img_size: int = 224,    
    # 32 as default for my pc's memory
    batch_size: int = 32,    
    # 0 = main process only - required for MPS 
    num_workers: int = 0,
    random_seed: int = 26,
    min_images: int = 50,
) -> tuple:
    

    print("\nBuilding DF20 datasets...")

    train_ds = DF20EdibilityDataset(
        metadata_csv, image_root,
        transform=get_train_transform(img_size),
        split='train', random_seed=random_seed, min_images=min_images,
    )
    val_ds = DF20EdibilityDataset(
        metadata_csv, image_root,
        # deterministic for val
        transform=get_val_transform(img_size),
        split='val', random_seed=random_seed, min_images=min_images,
    )
    test_ds = DF20EdibilityDataset(
        metadata_csv, image_root,
        # deterministic for test
        transform=get_val_transform(img_size),
        split='test', random_seed=random_seed, min_images=min_images,
    )

    # WeightedRandomSampler replaces shuffle=True for the training loader.
    # samples with replacement so that each class appears equally often,
    weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)

    print(f"\nDataLoaders ready:")
    print(f"Train: {len(train_ds):,} samples  "
          f"({len(train_loader)} batches @ batch_size={batch_size})")
    print(f"Val: {len(val_ds):,} samples")
    print(f"Test: {len(test_ds):,} samples")
    print(f"Species: {len(train_ds.species_list)}")

    return train_loader, val_loader, test_loader