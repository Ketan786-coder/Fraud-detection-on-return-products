"""
Image Quality Validation Utility
Validates images before fraud detection processing
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ImageQualityScore:
    """Image quality assessment results"""
    overall_score: float  # 0-1
    resolution_score: float
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    noise_score: float
    compression_score: float
    is_acceptable: bool
    feedback: List[str]


class ImageQualityValidator:
    """Validates image quality for fraud detection"""

    def __init__(self):
        self.min_resolution = 720  # Minimum height in pixels
        self.min_sharpness = 100   # Laplacian variance threshold
        self.min_brightness = 0.2
        self.max_brightness = 0.85
        self.min_contrast = 0.1
        self.min_file_size_kb = 50  # Minimum 50KB (avoids over-compression)

    def validate_image(self, image_path: str) -> Tuple[bool, ImageQualityScore]:
        """
        Validate image quality
        
        Returns:
            (is_acceptable, quality_score)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, self._create_score(0.0, [], False, ["Invalid image file"])
            
            return self.validate_array(image), self._create_score(0.0, [], False, [])
        except Exception as e:
            return False, self._create_score(0.0, [], False, [str(e)])

    def validate_array(self, image: np.ndarray) -> Tuple[bool, ImageQualityScore]:
        """Validate image array"""
        feedback = []
        h, w = image.shape[:2]

        # 1. Resolution check
        if h >= 1080:
            resolution_score = 1.0
            feedback.append("✓ Resolution excellent (1080p+)")
        elif h >= 720:
            resolution_score = 0.8
            feedback.append("✓ Resolution good (720p)")
        elif h >= 480:
            resolution_score = 0.5
            feedback.append("⚠ Resolution acceptable (480p) - use 720p+ for better accuracy")
        else:
            resolution_score = 0
            feedback.append("❌ Resolution too low (<480p)")

        # 2. Sharpness check (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var >= 2000:
            sharpness_score = 1.0
            feedback.append("✓ Sharpness excellent")
        elif laplacian_var >= 1000:
            sharpness_score = 0.8
            feedback.append("✓ Sharpness good")
        elif laplacian_var >= 100:
            sharpness_score = 0.5
            feedback.append("⚠ Image somewhat blurry - use tripod/stabilizer")
        else:
            sharpness_score = 0
            feedback.append("❌ Image too blurry - use tripod and proper focus")

        # 3. Brightness check
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]) / 255

        if self.min_brightness <= brightness <= self.max_brightness:
            brightness_score = 1.0
            feedback.append("✓ Brightness optimal")
        elif 0.15 <= brightness < self.min_brightness or self.max_brightness < brightness <= 0.9:
            brightness_score = 0.7
            feedback.append(f"⚠ Brightness suboptimal ({brightness:.0%}) - adjust lighting")
        else:
            brightness_score = 0
            feedback.append("❌ Image too dark or too bright - improve lighting")

        # 4. Contrast check
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        contrast = np.std(gray_norm)

        if contrast >= 0.3:
            contrast_score = 1.0
            feedback.append("✓ Contrast excellent")
        elif contrast >= 0.1:
            contrast_score = 0.8
            feedback.append("✓ Contrast good")
        else:
            contrast_score = 0.4
            feedback.append("⚠ Low contrast - improve lighting setup")

        # 5. Noise detection (variance between image and Gaussian blur)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.var(gray.astype(float) - blur.astype(float))

        if noise_level <= 50:
            noise_score = 1.0
            feedback.append("✓ Noise level acceptable")
        elif noise_level <= 200:
            noise_score = 0.8
            feedback.append("✓ Noise level low")
        else:
            noise_score = 0.5
            feedback.append("⚠ High noise detected - improve lighting or reduce ISO")

        # 6. Compression artifacts (edge density)
        edges = cv2.Canny(gray, 50, 150)
        artifact_ratio = np.sum(edges) / edges.size

        if artifact_ratio < 0.10:
            compression_score = 1.0
            feedback.append("✓ Compression quality good")
        elif artifact_ratio < 0.15:
            compression_score = 0.8
            feedback.append("✓ Compression acceptable")
        else:
            compression_score = 0.5
            feedback.append("⚠ Compression artifacts detected - use higher quality setting")

        # Calculate overall score (weighted)
        weights = {
            'resolution': 0.15,
            'sharpness': 0.25,
            'brightness': 0.20,
            'contrast': 0.15,
            'noise': 0.15,
            'compression': 0.10
        }

        overall_score = (
            weights['resolution'] * resolution_score +
            weights['sharpness'] * sharpness_score +
            weights['brightness'] * brightness_score +
            weights['contrast'] * contrast_score +
            weights['noise'] * noise_score +
            weights['compression'] * compression_score
        )

        # Determine acceptability (need high quality for fraud detection)
        is_acceptable = (
            resolution_score >= 0.5 and
            sharpness_score >= 0.5 and
            brightness_score >= 0.5 and
            contrast_score >= 0.5 and
            overall_score >= 0.6
        )

        if not is_acceptable:
            feedback.append("\n❌ IMAGE REJECTED - Retake with better conditions")
        else:
            feedback.append("\n✓ IMAGE ACCEPTED - Quality suitable for analysis")

        return is_acceptable, self._create_score(
            overall_score,
            [resolution_score, sharpness_score, brightness_score, 
             contrast_score, noise_score, compression_score],
            is_acceptable,
            feedback
        )

    def _create_score(self, overall: float, component_scores: List[float], 
                     acceptable: bool, feedback: List[str]) -> ImageQualityScore:
        """Create quality score object"""
        if component_scores:
            resolution_score, sharpness_score, brightness_score, contrast_score, noise_score, compression_score = component_scores
        else:
            resolution_score = sharpness_score = brightness_score = contrast_score = noise_score = compression_score = 0.0

        return ImageQualityScore(
            overall_score=overall,
            resolution_score=resolution_score,
            sharpness_score=sharpness_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            noise_score=noise_score,
            compression_score=compression_score,
            is_acceptable=acceptable,
            feedback=feedback
        )

    def validate_batch(self, image_paths: List[str]) -> Dict[str, Tuple[bool, ImageQualityScore]]:
        """Validate multiple images"""
        results = {}
        for path in image_paths:
            acceptable, score = self.validate_image(path)
            results[path] = (acceptable, score)
        return results

    def get_quality_report(self, image_path: str) -> str:
        """Generate human-readable quality report"""
        acceptable, score = self.validate_image(image_path)

        report = f"""
{'=' * 60}
IMAGE QUALITY REPORT
{'=' * 60}

File: {image_path}
Status: {'✓ ACCEPTABLE' if acceptable else '❌ REJECTED'}

Overall Score: {score.overall_score:.1%}

Component Scores:
  Resolution:   {score.resolution_score:.1%}
  Sharpness:    {score.sharpness_score:.1%}
  Brightness:   {score.brightness_score:.1%}
  Contrast:     {score.contrast_score:.1%}
  Noise:        {score.noise_score:.1%}
  Compression:  {score.compression_score:.1%}

Feedback:
{chr(10).join('  ' + line for line in score.feedback)}

{'=' * 60}
"""
        return report


# Command-line utility
if __name__ == "__main__":
    import sys

    validator = ImageQualityValidator()

    if len(sys.argv) < 2:
        print("Usage: python image_validator.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(validator.get_quality_report(image_path))
