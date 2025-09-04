"""
Data Loading Module for Oil Spill Detection
Handles loading and organizing satellite imagery datasets
"""

import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilSpillDataLoader:
    """
    Data loader class for oil spill detection dataset
    Supports both classification (train/val/test folders) and segmentation (image/mask pairs)
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader
        
        Args:
            data_dir (str): Path to the dataset directory
        """
        self.data_dir = r"E:\MyDownloads\dataset"
        
        self.dataset_type = self._detect_dataset_type()
        
        if self.dataset_type == 'classification':
            self.train_dir = os.path.join(data_dir, 'train')
            self.val_dir = os.path.join(data_dir, 'val') 
            self.test_dir = os.path.join(data_dir, 'test')
            self.label_colors_file = os.path.join(data_dir, 'label_colors')
        else:
            # Original segmentation structure
            self.images_dir = os.path.join(data_dir, 'images')
            self.masks_dir = os.path.join(data_dir, 'masks')
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.masks_dir, exist_ok=True)
        
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
    def _detect_dataset_type(self) -> str:
        """
        Detect if dataset is classification or segmentation type
        
        Returns:
            str: 'classification' or 'segmentation'
        """
        has_train_val_test = (
            os.path.exists(os.path.join(self.data_dir, 'train')) and
            os.path.exists(os.path.join(self.data_dir, 'val')) and
            os.path.exists(os.path.join(self.data_dir, 'test'))
        )
        
        if has_train_val_test:
            logger.info("Detected classification dataset structure (train/val/test folders)")
            return 'classification'
        else:
            logger.info("Using segmentation dataset structure (images/masks folders)")
            return 'segmentation'
    
    def load_dataset_info(self) -> Dict:
        """
        Load and return dataset information
        
        Returns:
            Dict: Dataset statistics and information
        """
        if self.dataset_type == 'classification':
            return self._load_classification_info()
        else:
            return self._load_segmentation_info()
    
    def _load_classification_info(self) -> Dict:
        """
        Load classification dataset information
        
        Returns:
            Dict: Dataset statistics and information
        """
        dataset_info = {
            'dataset_type': 'classification',
            'train_samples': 0,
            'val_samples': 0, 
            'test_samples': 0,
            'classes': [],
            'class_distribution': {}
        }
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        # Process each split
        for split_name, split_dir in [('train', self.train_dir), ('val', self.val_dir), ('test', self.test_dir)]:
            if not os.path.exists(split_dir):
                logger.warning(f"Directory not found: {split_dir}")
                continue
                
            split_samples = 0
            
            # Get class folders or files
            items = os.listdir(split_dir)
            
            # Check if there are class subdirectories
            class_dirs = [item for item in items if os.path.isdir(os.path.join(split_dir, item))]
            
            if class_dirs:
                # Multi-class structure with subdirectories
                for class_name in class_dirs:
                    class_dir = os.path.join(split_dir, class_name)
                    class_files = [f for f in os.listdir(class_dir) 
                                 if any(f.lower().endswith(ext) for ext in image_extensions)]
                    
                    split_samples += len(class_files)
                    
                    if class_name not in dataset_info['classes']:
                        dataset_info['classes'].append(class_name)
                    
                    if class_name not in dataset_info['class_distribution']:
                        dataset_info['class_distribution'][class_name] = 0
                    dataset_info['class_distribution'][class_name] += len(class_files)
            else:
                # All images in one folder - binary classification
                image_files = [f for f in items 
                             if any(f.lower().endswith(ext) for ext in image_extensions)]
                split_samples = len(image_files)
            
            dataset_info[f'{split_name}_samples'] = split_samples
        
        # Load label colors if available
        if os.path.exists(self.label_colors_file):
            try:
                with open(self.label_colors_file, 'r') as f:
                    label_info = f.read().strip()
                    dataset_info['label_colors_info'] = label_info
            except Exception as e:
                logger.warning(f"Could not read label_colors file: {e}")
        
        dataset_info['total_samples'] = (dataset_info['train_samples'] + 
                                       dataset_info['val_samples'] + 
                                       dataset_info['test_samples'])
        
        logger.info(f"Classification dataset loaded: {dataset_info['total_samples']} total samples")
        return dataset_info
    
    def _load_segmentation_info(self) -> Dict:
        """
        Load segmentation dataset information (original implementation)
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        for ext in image_extensions:
            self.image_paths.extend([
                os.path.join(self.images_dir, f) 
                for f in os.listdir(self.images_dir) 
                if f.lower().endswith(ext.lower())
            ])
        
        # Get corresponding mask files
        for img_path in self.image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Look for corresponding mask
            mask_path = None
            for ext in image_extensions:
                potential_mask = os.path.join(self.masks_dir, f"{img_name}{ext}")
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
            
            if mask_path:
                self.mask_paths.append(mask_path)
            else:
                logger.warning(f"No mask found for image: {img_path}")
        
        # Ensure equal number of images and masks
        min_length = min(len(self.image_paths), len(self.mask_paths))
        self.image_paths = self.image_paths[:min_length]
        self.mask_paths = self.mask_paths[:min_length]
        
        dataset_info = {
            'dataset_type': 'segmentation',
            'total_samples': len(self.image_paths),
            'images_dir': self.images_dir,
            'masks_dir': self.masks_dir,
            'sample_image_paths': self.image_paths[:5],
            'sample_mask_paths': self.mask_paths[:5]
        }
        
        logger.info(f"Segmentation dataset loaded: {dataset_info['total_samples']} samples")
        return dataset_info
    
    def load_classification_data(self, split: str = 'train', target_size: Tuple[int, int] = (256, 256)) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load classification data from train/val/test split
        
        Args:
            split (str): 'train', 'val', or 'test'
            target_size (Tuple[int, int]): Target size for resizing
            
        Returns:
            Tuple[List[np.ndarray], List[int]]: Images and labels
        """
        if self.dataset_type != 'classification':
            raise ValueError("This method only works with classification datasets")
        
        split_dir = getattr(self, f'{split}_dir')
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        images = []
        labels = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        items = os.listdir(split_dir)
        class_dirs = [item for item in items if os.path.isdir(os.path.join(split_dir, item))]
        
        if class_dirs:
            # Multi-class with subdirectories
            class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_dirs))}
            
            for class_name in class_dirs:
                class_dir = os.path.join(split_dir, class_name)
                class_files = [f for f in os.listdir(class_dir) 
                             if any(f.lower().endswith(ext) for ext in image_extensions)]
                
                for img_file in class_files:
                    img_path = os.path.join(class_dir, img_file)
                    
                    # Load and preprocess image
                    image = cv2.imread(img_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, target_size)
                        image = image.astype(np.float32) / 255.0
                        
                        images.append(image)
                        labels.append(class_to_idx[class_name])
        else:
            # Binary classification - all images in one folder
            # Assume oil spill detection: look for 'oil' or 'spill' in filename for label 1
            image_files = [f for f in items 
                         if any(f.lower().endswith(ext) for ext in image_extensions)]
            
            for img_file in image_files:
                img_path = os.path.join(split_dir, img_file)
                
                # Load and preprocess image
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, target_size)
                    image = image.astype(np.float32) / 255.0
                    
                    images.append(image)
                    
                    # Determine label from filename (basic heuristic)
                    filename_lower = img_file.lower()
                    if 'oil' in filename_lower or 'spill' in filename_lower:
                        labels.append(1)  # Oil spill
                    else:
                        labels.append(0)  # Clean water
        
        logger.info(f"Loaded {len(images)} images from {split} split")
        return images, labels

    def load_image_pair(self, idx: int, target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and mask pair (for segmentation datasets)
        """
        if self.dataset_type != 'segmentation':
            raise ValueError("This method only works with segmentation datasets")
            
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
        
        # Resize to target size
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        
        return image, mask
    
    def create_train_val_split(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Create train-validation split
        
        Args:
            test_size (float): Proportion of validation set
            random_state (int): Random seed for reproducibility
            
        Returns:
            Dict: Train and validation indices
        """
        if self.dataset_type != 'segmentation':
            raise ValueError("This method only works with segmentation datasets")
        
        indices = list(range(len(self.image_paths)))
        
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        split_info = {
            'train_indices': train_idx,
            'val_indices': val_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        }
        
        logger.info(f"Dataset split - Train: {len(train_idx)}, Val: {len(val_idx)}")
        return split_info
    
    def get_sample_statistics(self, num_samples: int = 100) -> Dict:
        """
        Calculate dataset statistics
        
        Args:
            num_samples (int): Number of samples to analyze
            
        Returns:
            Dict: Dataset statistics
        """
        if len(self.image_paths) == 0:
            return {}
        
        sample_size = min(num_samples, len(self.image_paths))
        
        image_shapes = []
        spill_ratios = []
        
        for i in range(sample_size):
            try:
                image, mask = self.load_image_pair(i)
                image_shapes.append(image.shape)
                
                # Calculate oil spill ratio
                spill_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
                spill_ratios.append(spill_ratio)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        stats = {
            'sample_count': sample_size,
            'average_spill_ratio': np.mean(spill_ratios) if spill_ratios else 0,
            'spill_ratio_std': np.std(spill_ratios) if spill_ratios else 0,
            'min_spill_ratio': np.min(spill_ratios) if spill_ratios else 0,
            'max_spill_ratio': np.max(spill_ratios) if spill_ratios else 0,
            'image_shapes': image_shapes[:10]  # First 10 shapes
        }
        
        return stats

def create_sample_dataset(output_dir: str, num_samples: int = 50):
    """
    Create a sample dataset for testing (if no real dataset is available)
    
    Args:
        output_dir (str): Directory to save sample data
        num_samples (int): Number of sample images to create
    """
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    logger.info(f"Creating {num_samples} sample images...")
    
    for i in range(num_samples):
        # Create synthetic satellite image (ocean with potential oil spill)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add ocean-like blue tint
        image[:, :, 2] = np.clip(image[:, :, 2] + 50, 0, 255)  # More blue
        image[:, :, 1] = np.clip(image[:, :, 1] + 20, 0, 255)  # Some green
        
        # Create corresponding mask with random oil spill regions
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Add random oil spill regions
        if np.random.random() > 0.3:  # 70% chance of having oil spill
            num_spills = np.random.randint(1, 4)
            for _ in range(num_spills):
                center_x = np.random.randint(50, 206)
                center_y = np.random.randint(50, 206)
                radius = np.random.randint(10, 30)
                
                y, x = np.ogrid[:256, :256]
                mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                mask[mask_circle] = 255
        
        # Save image and mask
        image_path = os.path.join(images_dir, f'sample_{i:03d}.png')
        mask_path = os.path.join(masks_dir, f'sample_{i:03d}.png')
        
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
    
    logger.info(f"Sample dataset created in {output_dir}")
    
    info_file = os.path.join(output_dir, 'DATASET_INFO.txt')
    with open(info_file, 'w') as f:
        f.write("SAMPLE DATASET CREATED\n")
        f.write("=" * 50 + "\n\n")
        f.write("This is a synthetic dataset created for testing purposes.\n")
        f.write("To use your own dataset:\n\n")
        f.write("1. Replace the contents of data/raw/images/ with your satellite images\n")
        f.write("2. Replace the contents of data/raw/masks/ with corresponding oil spill masks\n")
        f.write("3. Supported formats: .jpg, .jpeg, .png, .tif, .tiff\n")
        f.write("4. Ensure image and mask filenames match (except extension)\n\n")
        f.write("Dataset Sources:\n")
        f.write("- Kaggle: https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection\n")
        f.write("- ESA Copernicus: https://scihub.copernicus.eu/\n")
        f.write("- NASA Earth Data: https://earthdata.nasa.gov/\n")

if __name__ == "__main__":
    # Example usage
    data_dir = r"E:\MyDownloads\dataset"
    
    # Create sample dataset if no real data exists
    if not os.path.exists(data_dir):
        create_sample_dataset(data_dir, num_samples=100)
    
    # Load dataset
    loader = OilSpillDataLoader(data_dir)
    dataset_info = loader.load_dataset_info()
    
    print("Dataset Information:")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    
    # Load data based on dataset type
    if loader.dataset_type == 'classification':
        try:
            train_images, train_labels = loader.load_classification_data('train')
            print(f"\nLoaded {len(train_images)} training images")
            print(f"Label distribution: {np.bincount(train_labels)}")
        except Exception as e:
            print(f"Error loading classification data: {e}")
    else:
        # Get statistics for segmentation
        stats = loader.get_sample_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
