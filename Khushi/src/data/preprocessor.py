"""
Data Preprocessing Module for Oil Spill Detection
Handles image preprocessing, normalization, and preparation for training
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilSpillPreprocessor:
    """
    Preprocessing class for oil spill detection images
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize preprocessor
        
        Args:
            target_size (Tuple[int, int]): Target image size
        """
        self.target_size = target_size
        self.preprocessing_stats = {}
        
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image (np.ndarray): Input image
            method (str): Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            np.ndarray: Normalized image
        """
        if method == 'minmax':
            # Min-Max normalization to [0, 1]
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            image_norm = (image - mean) / (std + 1e-8)
        
        elif method == 'robust':
            # Robust normalization using percentiles
            p25, p75 = np.percentile(image, [25, 75])
            image_norm = (image - p25) / (p75 - p25 + 1e-8)
            image_norm = np.clip(image_norm, 0, 1)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return image_norm.astype(np.float32)
    
    def reduce_speckle_noise(self, image: np.ndarray, filter_type: str = 'gaussian') -> np.ndarray:
        """
        Reduce speckle noise in SAR images
        
        Args:
            image (np.ndarray): Input image
            filter_type (str): Filter type ('gaussian', 'median', 'bilateral')
            
        Returns:
            np.ndarray: Filtered image
        """
        if len(image.shape) == 3:
            # Convert to grayscale for filtering
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        if filter_type == 'gaussian':
            filtered = cv2.GaussianBlur(gray, (5, 5), 0)
        elif filter_type == 'median':
            filtered = cv2.medianBlur(gray, 5)
        elif filter_type == 'bilateral':
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Convert back to original format
        if len(image.shape) == 3:
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            return filtered_rgb.astype(np.float32) / 255.0
        else:
            return filtered.astype(np.float32) / 255.0
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast
        
        Args:
            image (np.ndarray): Input image
            method (str): Enhancement method ('clahe', 'histogram_eq')
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            if len(img_uint8.shape) == 3:
                # Apply CLAHE to each channel
                enhanced = np.zeros_like(img_uint8)
                for i in range(3):
                    enhanced[:, :, i] = clahe.apply(img_uint8[:, :, i])
            else:
                enhanced = clahe.apply(img_uint8)
        
        elif method == 'histogram_eq':
            # Standard histogram equalization
            if len(img_uint8.shape) == 3:
                enhanced = np.zeros_like(img_uint8)
                for i in range(3):
                    enhanced[:, :, i] = cv2.equalizeHist(img_uint8[:, :, i])
            else:
                enhanced = cv2.equalizeHist(img_uint8)
        
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
        
        return enhanced.astype(np.float32) / 255.0
    
    def preprocess_image(self, 
                        image: np.ndarray, 
                        apply_noise_reduction: bool = True,
                        apply_contrast_enhancement: bool = True,
                        normalization_method: str = 'minmax') -> np.ndarray:
        """
        Complete preprocessing pipeline for a single image
        
        Args:
            image (np.ndarray): Input image
            apply_noise_reduction (bool): Whether to apply noise reduction
            apply_contrast_enhancement (bool): Whether to enhance contrast
            normalization_method (str): Normalization method
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize image
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size)
        
        # Apply noise reduction for SAR images
        if apply_noise_reduction:
            image = self.reduce_speckle_noise(image, filter_type='gaussian')
        
        # Enhance contrast
        if apply_contrast_enhancement:
            image = self.enhance_contrast(image, method='clahe')
        
        # Normalize pixel values
        image = self.normalize_image(image, method=normalization_method)
        
        return image
    
    def preprocess_mask(self, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Preprocess segmentation mask
        
        Args:
            mask (np.ndarray): Input mask
            threshold (float): Binarization threshold
            
        Returns:
            np.ndarray: Preprocessed mask
        """
        # Resize mask
        if mask.shape[:2] != self.target_size:
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Ensure single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Binarize
        mask = (mask > threshold).astype(np.float32)
        
        return mask
    
    def calculate_dataset_statistics(self, 
                                   image_paths: List[str], 
                                   mask_paths: List[str]) -> Dict:
        """
        Calculate comprehensive dataset statistics
        
        Args:
            image_paths (List[str]): List of image file paths
            mask_paths (List[str]): List of mask file paths
            
        Returns:
            Dict: Dataset statistics
        """
        logger.info("Calculating dataset statistics...")
        
        pixel_values = []
        spill_ratios = []
        image_sizes = []
        
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), 
                                       total=len(image_paths),
                                       desc="Processing images"):
            try:
                # Load image and mask
                image = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None or mask is None:
                    continue
                
                # Convert image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Store original size
                image_sizes.append(image.shape[:2])
                
                # Preprocess
                image = self.preprocess_image(image)
                mask = self.preprocess_mask(mask)
                
                # Collect pixel values
                pixel_values.extend(image.flatten())
                
                # Calculate spill ratio
                spill_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
                spill_ratios.append(spill_ratio)
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate statistics
        pixel_values = np.array(pixel_values)
        spill_ratios = np.array(spill_ratios)
        
        stats = {
            'total_samples': len(image_paths),
            'processed_samples': len(spill_ratios),
            'pixel_statistics': {
                'mean': float(np.mean(pixel_values)),
                'std': float(np.std(pixel_values)),
                'min': float(np.min(pixel_values)),
                'max': float(np.max(pixel_values)),
                'median': float(np.median(pixel_values))
            },
            'spill_statistics': {
                'mean_spill_ratio': float(np.mean(spill_ratios)),
                'std_spill_ratio': float(np.std(spill_ratios)),
                'min_spill_ratio': float(np.min(spill_ratios)),
                'max_spill_ratio': float(np.max(spill_ratios)),
                'samples_with_spill': int(np.sum(spill_ratios > 0)),
                'samples_without_spill': int(np.sum(spill_ratios == 0))
            },
            'image_size_distribution': {
                'unique_sizes': list(set(image_sizes)),
                'most_common_size': max(set(image_sizes), key=image_sizes.count) if image_sizes else None
            }
        }
        
        self.preprocessing_stats = stats
        return stats
    
    def visualize_preprocessing_effects(self, 
                                      image: np.ndarray, 
                                      save_path: Optional[str] = None) -> None:
        """
        Visualize the effects of different preprocessing steps
        
        Args:
            image (np.ndarray): Input image
            save_path (Optional[str]): Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Noise reduction
        denoised = self.reduce_speckle_noise(image)
        axes[0, 1].imshow(denoised)
        axes[0, 1].set_title('Noise Reduction')
        axes[0, 1].axis('off')
        
        # Contrast enhancement
        enhanced = self.enhance_contrast(image)
        axes[0, 2].imshow(enhanced)
        axes[0, 2].set_title('Contrast Enhancement')
        axes[0, 2].axis('off')
        
        # Normalized
        normalized = self.normalize_image(image)
        axes[1, 0].imshow(normalized)
        axes[1, 0].set_title('Normalized')
        axes[1, 0].axis('off')
        
        # Full preprocessing
        preprocessed = self.preprocess_image(image)
        axes[1, 1].imshow(preprocessed)
        axes[1, 1].set_title('Full Preprocessing')
        axes[1, 1].axis('off')
        
        # Histogram comparison
        axes[1, 2].hist(image.flatten(), bins=50, alpha=0.7, label='Original', density=True)
        axes[1, 2].hist(preprocessed.flatten(), bins=50, alpha=0.7, label='Preprocessed', density=True)
        axes[1, 2].set_title('Pixel Value Distribution')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Preprocessing visualization saved to {save_path}")
        
        plt.show()
    
    def save_preprocessing_stats(self, output_path: str) -> None:
        """
        Save preprocessing statistics to JSON file
        
        Args:
            output_path (str): Path to save statistics
        """
        if not self.preprocessing_stats:
            logger.warning("No preprocessing statistics available to save")
            return
        
        with open(output_path, 'w') as f:
            json.dump(self.preprocessing_stats, f, indent=2)
        
        logger.info(f"Preprocessing statistics saved to {output_path}")

def create_preprocessing_report(stats: Dict, output_dir: str) -> None:
    """
    Create a comprehensive preprocessing report
    
    Args:
        stats (Dict): Preprocessing statistics
        output_dir (str): Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Spill ratio distribution
    spill_with = stats['spill_statistics']['samples_with_spill']
    spill_without = stats['spill_statistics']['samples_without_spill']
    
    axes[0, 0].pie([spill_with, spill_without], 
                   labels=['With Oil Spill', 'Without Oil Spill'],
                   autopct='%1.1f%%',
                   startangle=90)
    axes[0, 0].set_title('Dataset Distribution')
    
    # Pixel value statistics
    pixel_stats = stats['pixel_statistics']
    metrics = ['mean', 'std', 'min', 'max', 'median']
    values = [pixel_stats[metric] for metric in metrics]
    
    axes[0, 1].bar(metrics, values)
    axes[0, 1].set_title('Pixel Value Statistics')
    axes[0, 1].set_ylabel('Value')
    
    # Spill ratio statistics
    spill_stats = stats['spill_statistics']
    spill_metrics = ['mean_spill_ratio', 'std_spill_ratio', 'min_spill_ratio', 'max_spill_ratio']
    spill_values = [spill_stats[metric] for metric in spill_metrics]
    
    axes[1, 0].bar([m.replace('_spill_ratio', '') for m in spill_metrics], spill_values)
    axes[1, 0].set_title('Oil Spill Ratio Statistics')
    axes[1, 0].set_ylabel('Ratio')
    
    # Processing summary
    axes[1, 1].text(0.1, 0.8, f"Total Samples: {stats['total_samples']}", fontsize=12)
    axes[1, 1].text(0.1, 0.7, f"Processed: {stats['processed_samples']}", fontsize=12)
    axes[1, 1].text(0.1, 0.6, f"Success Rate: {stats['processed_samples']/stats['total_samples']*100:.1f}%", fontsize=12)
    axes[1, 1].text(0.1, 0.4, f"Samples with Spill: {spill_with}", fontsize=12)
    axes[1, 1].text(0.1, 0.3, f"Samples without Spill: {spill_without}", fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Processing Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preprocessing_report.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Preprocessing report saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    from data_loader import OilSpillDataLoader
    
    # Initialize preprocessor
    preprocessor = OilSpillPreprocessor(target_size=(256, 256))
    
    # Load dataset
    data_dir = "E:\MyDownloads\dataset"
    loader = OilSpillDataLoader(data_dir)
    dataset_info = loader.load_dataset_info()
    
    if dataset_info['total_samples'] > 0:
        # Calculate preprocessing statistics
        stats = preprocessor.calculate_dataset_statistics(
            loader.image_paths, 
            loader.mask_paths
        )
        
        # Save statistics
        os.makedirs("results", exist_ok=True)
        preprocessor.save_preprocessing_stats("results/preprocessing_stats.json")
        
        # Create report
        create_preprocessing_report(stats, "results/figures")
        
        # Visualize preprocessing on sample image
        sample_image, _ = loader.load_image_pair(0)
        preprocessor.visualize_preprocessing_effects(
            sample_image, 
            "results/figures/preprocessing_effects.png"
        )
        
        print("Preprocessing analysis complete!")
    else:
        print("No dataset found. Please add your dataset to the E:\MyDownloads\dataset directory.")
