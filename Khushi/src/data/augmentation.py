"""
Data Augmentation Module for Oil Spill Detection
Implements various augmentation techniques to increase dataset diversity
"""

import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilSpillAugmentor:
    """
    Data augmentation class for oil spill detection
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize augmentor
        
        Args:
            target_size (Tuple[int, int]): Target image size
        """
        self.target_size = target_size
        self.augmentation_stats = {}
        
        # Define augmentation pipelines
        self.train_transforms = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.3
            ),
            
            # Photometric transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.1),
            
            # Weather and atmospheric effects
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=1,
                num_flare_circles_upper=2,
                p=0.05
            ),
            
            # Elastic transformations
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.1
            ),
            
            # Grid distortion
            A.GridDistortion(p=0.1),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.1),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.validation_transforms = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Augmentation for satellite imagery specific
        self.sar_transforms = A.Compose([
            # SAR-specific augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.4),
            
            # Speckle noise simulation
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            
            # Intensity variations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.4
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            
            # Atmospheric effects
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.1),
            
            A.Normalize(mean=[0.5], std=[0.5]),  # For single channel SAR
        ])
    
    def augment_image_pair(self, 
                          image: np.ndarray, 
                          mask: np.ndarray,
                          transform_type: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to image-mask pair
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Input mask
            transform_type (str): Type of transformation ('train', 'val', 'sar')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented image and mask
        """
        # Ensure proper format
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Select appropriate transform
        if transform_type == 'train':
            transform = self.train_transforms
        elif transform_type == 'val':
            transform = self.validation_transforms
        elif transform_type == 'sar':
            transform = self.sar_transforms
            if image.shape[-1] == 3:
                # Convert to grayscale for SAR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=-1)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Apply transformation
        transformed = transform(image=image, mask=mask)
        
        return transformed['image'], transformed['mask']
    
    def create_augmented_dataset(self,
                               image_paths: List[str],
                               mask_paths: List[str],
                               output_dir: str,
                               augmentations_per_image: int = 5,
                               transform_type: str = 'train') -> Dict:
        """
        Create augmented dataset
        
        Args:
            image_paths (List[str]): List of original image paths
            mask_paths (List[str]): List of original mask paths
            output_dir (str): Output directory for augmented data
            augmentations_per_image (int): Number of augmentations per original image
            transform_type (str): Type of transformation to apply
            
        Returns:
            Dict: Augmentation statistics
        """
        # Create output directories
        aug_images_dir = os.path.join(output_dir, 'images')
        aug_masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(aug_images_dir, exist_ok=True)
        os.makedirs(aug_masks_dir, exist_ok=True)
        
        logger.info(f"Creating augmented dataset with {augmentations_per_image} augmentations per image...")
        
        total_generated = 0
        failed_augmentations = 0
        
        for idx, (img_path, mask_path) in enumerate(tqdm(
            zip(image_paths, mask_paths),
            total=len(image_paths),
            desc="Augmenting images"
        )):
            try:
                # Load original image and mask
                image = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None or mask is None:
                    logger.warning(f"Could not load {img_path} or {mask_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                image = cv2.resize(image, self.target_size)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                
                # Normalize image
                image = image.astype(np.float32) / 255.0
                mask = (mask > 127).astype(np.uint8)
                
                # Generate augmentations
                for aug_idx in range(augmentations_per_image):
                    try:
                        # Apply augmentation
                        aug_image, aug_mask = self.augment_image_pair(
                            image.copy(), 
                            mask.copy(), 
                            transform_type
                        )
                        
                        # Convert back to uint8 for saving
                        if transform_type in ['train', 'val']:
                            # Denormalize
                            aug_image = aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                            aug_image = np.clip(aug_image, 0, 1)
                        
                        aug_image_uint8 = (aug_image * 255).astype(np.uint8)
                        aug_mask_uint8 = (aug_mask * 255).astype(np.uint8)
                        
                        # Save augmented image and mask
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        aug_img_name = f"{base_name}_aug_{aug_idx:02d}.png"
                        aug_mask_name = f"{base_name}_aug_{aug_idx:02d}.png"
                        
                        aug_img_path = os.path.join(aug_images_dir, aug_img_name)
                        aug_mask_path = os.path.join(aug_masks_dir, aug_mask_name)
                        
                        # Handle different image formats
                        if len(aug_image_uint8.shape) == 3:
                            cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image_uint8, cv2.COLOR_RGB2BGR))
                        else:
                            cv2.imwrite(aug_img_path, aug_image_uint8)
                        
                        cv2.imwrite(aug_mask_path, aug_mask_uint8)
                        
                        total_generated += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to augment {img_path}, augmentation {aug_idx}: {e}")
                        failed_augmentations += 1
                        continue
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate statistics
        stats = {
            'original_images': len(image_paths),
            'target_augmentations': len(image_paths) * augmentations_per_image,
            'generated_augmentations': total_generated,
            'failed_augmentations': failed_augmentations,
            'success_rate': total_generated / (len(image_paths) * augmentations_per_image) * 100,
            'output_directory': output_dir,
            'transform_type': transform_type
        }
        
        self.augmentation_stats = stats
        
        logger.info(f"Augmentation complete! Generated {total_generated} augmented samples")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        
        return stats
    
    def visualize_augmentations(self,
                              image: np.ndarray,
                              mask: np.ndarray,
                              num_examples: int = 6,
                              transform_type: str = 'train',
                              save_path: Optional[str] = None) -> None:
        """
        Visualize different augmentations applied to a sample
        
        Args:
            image (np.ndarray): Sample image
            mask (np.ndarray): Sample mask
            num_examples (int): Number of augmentation examples to show
            transform_type (str): Type of transformation
            save_path (Optional[str]): Path to save visualization
        """
        fig, axes = plt.subplots(3, num_examples + 1, figsize=(20, 12))
        
        # Show original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Original Mask')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = image.copy()
        if len(overlay.shape) == 3:
            overlay[:, :, 0] = np.where(mask > 0.5, 1.0, overlay[:, :, 0])
        axes[2, 0].imshow(overlay)
        axes[2, 0].set_title('Original Overlay')
        axes[2, 0].axis('off')
        
        # Generate and show augmentations
        for i in range(num_examples):
            try:
                aug_image, aug_mask = self.augment_image_pair(
                    image.copy(), 
                    mask.copy(), 
                    transform_type
                )
                
                # Denormalize for visualization if needed
                if transform_type in ['train', 'val']:
                    aug_image = aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    aug_image = np.clip(aug_image, 0, 1)
                
                # Show augmented image
                axes[0, i + 1].imshow(aug_image)
                axes[0, i + 1].set_title(f'Augmented {i + 1}')
                axes[0, i + 1].axis('off')
                
                # Show augmented mask
                axes[1, i + 1].imshow(aug_mask, cmap='gray')
                axes[1, i + 1].set_title(f'Aug Mask {i + 1}')
                axes[1, i + 1].axis('off')
                
                # Show overlay
                overlay = aug_image.copy()
                if len(overlay.shape) == 3:
                    overlay[:, :, 0] = np.where(aug_mask > 0.5, 1.0, overlay[:, :, 0])
                axes[2, i + 1].imshow(overlay)
                axes[2, i + 1].set_title(f'Aug Overlay {i + 1}')
                axes[2, i + 1].axis('off')
                
            except Exception as e:
                logger.warning(f"Failed to generate augmentation {i + 1}: {e}")
                continue
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Augmentation visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_augmentation_diversity(self,
                                     original_images: List[np.ndarray],
                                     augmented_images: List[np.ndarray]) -> Dict:
        """
        Analyze the diversity introduced by augmentations
        
        Args:
            original_images (List[np.ndarray]): List of original images
            augmented_images (List[np.ndarray]): List of augmented images
            
        Returns:
            Dict: Diversity analysis results
        """
        logger.info("Analyzing augmentation diversity...")
        
        # Calculate pixel value statistics
        orig_pixels = np.concatenate([img.flatten() for img in original_images])
        aug_pixels = np.concatenate([img.flatten() for img in augmented_images])
        
        diversity_stats = {
            'original_stats': {
                'mean': float(np.mean(orig_pixels)),
                'std': float(np.std(orig_pixels)),
                'min': float(np.min(orig_pixels)),
                'max': float(np.max(orig_pixels))
            },
            'augmented_stats': {
                'mean': float(np.mean(aug_pixels)),
                'std': float(np.std(aug_pixels)),
                'min': float(np.min(aug_pixels)),
                'max': float(np.max(aug_pixels))
            },
            'diversity_metrics': {
                'std_increase': float(np.std(aug_pixels) / np.std(orig_pixels)),
                'range_increase': float((np.max(aug_pixels) - np.min(aug_pixels)) / 
                                      (np.max(orig_pixels) - np.min(orig_pixels)))
            }
        }
        
        return diversity_stats
    
    def save_augmentation_stats(self, output_path: str) -> None:
        """
        Save augmentation statistics to JSON file
        
        Args:
            output_path (str): Path to save statistics
        """
        if not self.augmentation_stats:
            logger.warning("No augmentation statistics available to save")
            return
        
        with open(output_path, 'w') as f:
            json.dump(self.augmentation_stats, f, indent=2)
        
        logger.info(f"Augmentation statistics saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    from data_loader import OilSpillDataLoader
    
    # Initialize augmentor
    augmentor = OilSpillAugmentor(target_size=(256, 256))
    
    # Load dataset
    data_dir = "data/raw"
    loader = OilSpillDataLoader(data_dir)
    dataset_info = loader.load_dataset_info()
    
    if dataset_info['total_samples'] > 0:
        # Create augmented dataset
        output_dir = "data/augmented"
        stats = augmentor.create_augmented_dataset(
            loader.image_paths[:10],  # Use first 10 images for demo
            loader.mask_paths[:10],
            output_dir,
            augmentations_per_image=3
        )
        
        # Save statistics
        os.makedirs("results", exist_ok=True)
        augmentor.save_augmentation_stats("results/augmentation_stats.json")
        
        # Visualize augmentations on sample
        sample_image, sample_mask = loader.load_image_pair(0)
        augmentor.visualize_augmentations(
            sample_image,
            sample_mask,
            num_examples=5,
            save_path="results/figures/augmentation_examples.png"
        )
        
        print("Augmentation analysis complete!")
    else:
        print("No dataset found. Please add your dataset to the data/raw directory.")
