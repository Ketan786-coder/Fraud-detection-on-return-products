"""
AI-Based Product Return Fraud Detection System
Core Fraud Detection Engine

Author: Fraud Detection Team
Version: 1.0
License: MIT
"""

import cv2
import numpy as np
import pytesseract
from scipy.ndimage import laplace
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ComponentScores:
    """Individual component fraud scores"""
    ocr_score: float
    accessory_score: float
    damage_score: float
    swap_score: float
    wear_score: float


@dataclass
class FraudAnalysisResult:
    """Final fraud analysis result"""
    fraud_risk_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    component_scores: ComponentScores
    primary_fraud_type: Optional[str]
    confidence: float
    evidence: Dict
    timestamp: str
    recommendation: str


class ImageNormalizer:
    """Image preprocessing & normalization pipeline"""

    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
        """CLAHE + Gamma correction for lighting normalization"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel_enhanced = clahe.apply(l_channel)

        # Gamma correction
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        l_channel_enhanced = cv2.LUT(l_channel_enhanced, table)

        lab[:, :, 0] = l_channel_enhanced
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def white_balance(image: np.ndarray) -> np.ndarray:
        """Gray world assumption for white balance correction"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])

        result[:, :, 1] -= ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] -= ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Bilateral filtering for noise reduction (preserves edges)"""
        return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def standardize_resolution(image: np.ndarray, target_height: int = 1080) -> np.ndarray:
        """Standardize image resolution"""
        h, w = image.shape[:2]
        if h != target_height:
            aspect_ratio = w / h
            new_width = int(target_height * aspect_ratio)
            image = cv2.resize(image, (new_width, target_height),
                              interpolation=cv2.INTER_LANCZOS4)
        return image

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Complete normalization pipeline"""
        image = ImageNormalizer.normalize_lighting(image)
        image = ImageNormalizer.white_balance(image)
        image = ImageNormalizer.denoise(image)
        image = ImageNormalizer.standardize_resolution(image)
        return image


class OCRTextVerification:
    """OCR-based text extraction & verification"""

    @staticmethod
    def extract_text(image: np.ndarray) -> str:
        """Extract text using Tesseract OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-. '
        try:
            text = pytesseract.image_to_string(thresh, config=config)
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    @staticmethod
    def extract_serial_number(ocr_text: str) -> Dict:
        """Pattern matching for serial numbers"""
        import re

        patterns = {
            'serial': r'\b[A-Z]{1,3}\d{8,15}[A-Z]?\b',
            'imei': r'\b\d{15}\b',
            'model': r'\b[A-Z]+\d{2,4}[A-Z]?\b',
            'mac_address': r'\b([0-9A-Fa-f]{2}:){5}([0-9A-Fa-f]{2})\b',
        }

        results = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, ocr_text)
            results[key] = matches[0] if matches else None

        return results

    @staticmethod
    def compare_serial_numbers(delivery_text: str, return_text: str) -> Tuple[float, str, bool]:
        """Compare serial numbers between delivery and return
        
        Logic:
        - If serial in delivery but NOT in return → Product Swap (100%)
        - If serial in both and DIFFERENT → Product Swap (100%)
        - If serial in both and SAME → Same Product (0%)
        - If NO serial in delivery → Skip check (0%)
        
        Returns:
            Tuple of (score, evidence, serial_present_in_delivery)
        """
        delivery_serial = OCRTextVerification.extract_serial_number(delivery_text)
        return_serial = OCRTextVerification.extract_serial_number(return_text)

        # Check if serial was present in DELIVERY (most important)
        serial_in_delivery = bool(
            delivery_serial['imei'] or delivery_serial['serial']
        )

        # If NO serial in delivery, skip this check entirely
        if not serial_in_delivery:
            return 0.0, "No serial number in delivery image - check skipped", False
        
        # Serial WAS present in delivery - now check return
        serial_in_return = bool(
            return_serial['imei'] or return_serial['serial']
        )

        # IMEI comparison (most authoritative)
        if delivery_serial['imei']:
            if return_serial['imei']:
                # Both have IMEI
                if delivery_serial['imei'] == return_serial['imei']:
                    return 0.0, "IMEI MATCH - Same Product", True
                else:
                    return 100.0, "IMEI MISMATCH - PRODUCT SWAP DETECTED", True
            else:
                # Delivery has IMEI but return doesn't
                return 100.0, "IMEI in delivery but missing in return - PRODUCT SWAP", True

        # Serial number comparison (fallback if no IMEI)
        if delivery_serial['serial']:
            if return_serial['serial']:
                # Both have serial
                if delivery_serial['serial'] == return_serial['serial']:
                    return 0.0, "Serial MATCH - Same Product", True
                else:
                    return 100.0, "Serial MISMATCH - PRODUCT SWAP DETECTED", True
            else:
                # Delivery has serial but return doesn't
                return 100.0, "Serial in delivery but missing in return - PRODUCT SWAP", True
        
        return 0.0, "Serial verification inconclusive", serial_in_delivery


class AccessoryDetection:
    """Accessory detection & verification"""

    @staticmethod
    def detect_accessories(image: np.ndarray) -> Dict:
        """Detect accessories in image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for common accessories
        colors = {
            'black': [(0, 0, 0), (180, 255, 50)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'red': [(0, 100, 100), (10, 255, 255)],
            'blue': [(100, 100, 100), (130, 255, 255)],
            'silver': [(0, 0, 100), (180, 50, 200)],
        }

        detected = {}
        for color_name, (lower, upper) in colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected[color_name] = len(contours)

        return detected

    @staticmethod
    def compare_accessories(delivery_img: np.ndarray, return_img: np.ndarray) -> Tuple[float, str, bool]:
        """Compare accessories between delivery and return
        
        Returns:
            Tuple of (score, evidence, accessories_present_in_delivery)
        """
        delivery_acc = AccessoryDetection.detect_accessories(delivery_img)
        return_acc = AccessoryDetection.detect_accessories(return_img)

        # Calculate total items in delivery
        total_delivery = sum(delivery_acc.values())
        
        # If NO accessories in delivery image, skip this check
        if total_delivery == 0:
            return 0.0, "No accessories detected in delivery image - check skipped", False

        # Accessories WERE present in delivery - NOW CHECK if they're in return
        missing_items = 0
        for color, count in delivery_acc.items():
            if return_acc.get(color, 0) < count:
                missing_items += (count - return_acc.get(color, 0))

        # Score: 5 points per missing item
        accessory_score = min(100, missing_items * 5)

        if missing_items == 0:
            return 0.0, "All accessories present", True
        elif missing_items == 1:
            return 20.0, "1 accessory missing", True
        else:
            return accessory_score, f"{missing_items} accessories missing", True


class VisualConditionAnalysis:
    """Multi-layer visual condition analysis"""

    @staticmethod
    def edge_analysis(delivery_img: np.ndarray, return_img: np.ndarray) -> Tuple[float, str]:
        """Canny edge detection for damage detection"""
        delivery_gray = cv2.cvtColor(delivery_img, cv2.COLOR_BGR2GRAY)
        return_gray = cv2.cvtColor(return_img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        delivery_edges = cv2.Canny(delivery_gray, 50, 150)
        return_edges = cv2.Canny(return_gray, 50, 150)

        # Calculate edge density
        delivery_edge_density = np.sum(delivery_edges) / delivery_edges.size
        return_edge_density = np.sum(return_edges) / return_edges.size

        # Difference indicates new damage
        edge_difference = return_edge_density - delivery_edge_density

        if edge_difference < 0.01:
            return 0.0, "No new edge damage detected"
        elif edge_difference < 0.03:
            return 30.0, "Minor edge damage detected"
        else:
            return 60.0, "Significant edge damage detected"

    @staticmethod
    def keypoint_fingerprinting(delivery_img: np.ndarray, return_img: np.ndarray) -> Tuple[float, str]:
        """SIFT keypoint matching for product identity verification"""
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        delivery_kp, delivery_des = sift.detectAndCompute(delivery_img, None)
        return_kp, return_des = sift.detectAndCompute(return_img, None)

        if delivery_des is None or return_des is None:
            return 50.0, "Insufficient keypoints for identity verification"

        # Match keypoints
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(delivery_des, return_des, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Calculate match percentage
        match_percentage = len(good_matches) / max(len(delivery_kp), len(return_kp))

        if match_percentage > 0.8:
            return 0.0, f"Product identity confirmed ({match_percentage:.1%} keypoint match)"
        elif match_percentage > 0.5:
            return 50.0, f"Possible product swap ({match_percentage:.1%} keypoint match)"
        else:
            return 100.0, f"Definite product swap - different product ({match_percentage:.1%} keypoint match)"

    @staticmethod
    def texture_analysis(delivery_img: np.ndarray, return_img: np.ndarray) -> Tuple[float, str]:
        """LBP & Haralick texture analysis for wear detection"""
        delivery_gray = cv2.cvtColor(delivery_img, cv2.COLOR_BGR2GRAY)
        return_gray = cv2.cvtColor(return_img, cv2.COLOR_BGR2GRAY)

        # Local Binary Pattern
        lbp_delivery = local_binary_pattern(delivery_gray, 8, 1, method='uniform')
        lbp_return = local_binary_pattern(return_gray, 8, 1, method='uniform')

        # Haralick features
        glcm_delivery = graycomatrix(delivery_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
        glcm_return = graycomatrix(return_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

        # Contrast (higher = more worn)
        contrast_delivery = graycoprops(glcm_delivery, 'contrast').mean()
        contrast_return = graycoprops(glcm_return, 'contrast').mean()

        # Homogeneity (lower = more worn)
        homogeneity_delivery = graycoprops(glcm_delivery, 'homogeneity').mean()
        homogeneity_return = graycoprops(glcm_return, 'homogeneity').mean()

        wear_indicator = (contrast_return - contrast_delivery) - (homogeneity_delivery - homogeneity_return)

        if wear_indicator < 0.1:
            return 0.0, "No significant wear detected"
        elif wear_indicator < 0.3:
            return 20.0, "Minor wear detected"
        else:
            return 50.0, "Significant wear patterns detected (used product)"

    @staticmethod
    def high_frequency_analysis(delivery_img: np.ndarray, return_img: np.ndarray) -> Tuple[float, str]:
        """Wavelet decomposition for micro-scratch detection"""
        delivery_gray = cv2.cvtColor(delivery_img, cv2.COLOR_BGR2GRAY)
        return_gray = cv2.cvtColor(return_img, cv2.COLOR_BGR2GRAY)

        # Laplacian (approximates high-frequency content)
        delivery_laplacian = cv2.Laplacian(delivery_gray, cv2.CV_64F)
        return_laplacian = cv2.Laplacian(return_gray, cv2.CV_64F)

        # Energy in high-frequency bands
        delivery_hf_energy = np.sum(np.abs(delivery_laplacian))
        return_hf_energy = np.sum(np.abs(return_laplacian))

        # Ratio indicates new micro-damage
        if delivery_hf_energy > 0:
            hf_ratio = return_hf_energy / delivery_hf_energy
        else:
            hf_ratio = 1.0

        if hf_ratio < 1.2:
            return 0.0, "No micro-scratches detected"
        elif hf_ratio < 1.5:
            return 15.0, "Minor micro-scratches detected"
        else:
            return 40.0, "Multiple micro-scratches detected"


class RiskScoringEngine:
    """Weighted risk scoring & fraud classification"""

    # Default weights (can be adjusted per product category)
    # Priority: Product Identity > Condition
    WEIGHTS = {
        'ocr': 0.25,        # Serial mismatch = different product
        'swap': 0.40,       # Product swap = CRITICAL (different product)
        'damage': 0.20,     # Physical damage
        'accessory': 0.10,  # Missing accessories (only if product has them)
        'wear': 0.05        # Normal wear (least important)
    }

    @staticmethod
    def calculate_fraud_score(component_scores: ComponentScores) -> float:
        """Calculate fraud risk score
        
        LOGIC:
        - If critical fraud detected (serial/swap = 100) → fraud_score = 100
        - Otherwise → use weighted calculation
        """
        # PRIORITY CHECK: If critical product identity fraud detected, fraud score = 100
        if component_scores.ocr_score >= 100:
            return 100.0  # Serial mismatch = 100% fraud
        
        if component_scores.swap_score >= 100:
            return 100.0  # Product swap = 100% fraud
        
        if component_scores.accessory_score >= 100:
            return 100.0  # Critical accessories missing = 100% fraud
        
        if component_scores.damage_score >= 100:
            return 100.0  # Severe intentional damage = 100% fraud
        
        # Otherwise, use weighted calculation
        score = (
            RiskScoringEngine.WEIGHTS['ocr'] * component_scores.ocr_score +
            RiskScoringEngine.WEIGHTS['accessory'] * component_scores.accessory_score +
            RiskScoringEngine.WEIGHTS['damage'] * component_scores.damage_score +
            RiskScoringEngine.WEIGHTS['swap'] * component_scores.swap_score +
            RiskScoringEngine.WEIGHTS['wear'] * component_scores.wear_score
        )
        return min(100, max(0, score))

    @staticmethod
    def determine_risk_level(score: float) -> str:
        """Determine risk level from score"""
        if score < 20:
            return "LOW"
        elif score < 40:
            return "MEDIUM-LOW"
        elif score < 60:
            return "MEDIUM"
        elif score < 80:
            return "MEDIUM-HIGH"
        else:
            return "HIGH"

    @staticmethod
    def determine_recommendation(score: float, component_scores: ComponentScores = None) -> str:
        """Determine action recommendation using CASCADING LOGIC
        
        STEP 1: SERIAL NUMBER (HIGHEST PRIORITY)
                If serial mismatch → AUTO-DENY (100%) - PRODUCT SWAP CONFIRMED
        
        STEP 2: PRODUCT SWAP DETECTION (if no serial mismatch)
                If high visual mismatch → AUTO-DENY - Different product
        
        STEP 3: ACCESSORIES (only if no product swap)
                If accessories shown in delivery but missing in return → FRAUD
        
        STEP 4: DAMAGE/WEAR/TEAR (only if PRODUCT IS VERIFIED AS SAME)
                Check for intentional damage, wear patterns, usage signs
        """
        if not component_scores:
            # Default behavior if no component scores
            if score < 20:
                return "AUTO-APPROVE"
            elif score < 40:
                return "LIKELY APPROVE"
            elif score < 60:
                return "MANUAL REVIEW NEEDED"
            elif score < 80:
                return "LIKELY DENY"
            else:
                return "AUTO-DENY"
        
        # ========== STEP 1: SERIAL NUMBER CHECK (HIGHEST PRIORITY) ==========
        # Serial mismatch = DEFINITE PRODUCT SWAP
        if component_scores.ocr_score >= 100:
            return "AUTO-DENY"  # Serial mismatch = Different product
        elif component_scores.ocr_score >= 80:
            return "AUTO-DENY"  # Serial partially mismatch
        
        # ========== STEP 2: PRODUCT SWAP (if no serial provided) ==========
        # Multiple factors indicate different product
        if component_scores.swap_score >= 100:
            return "AUTO-DENY"  # Definite product swap (visual confirmation)
        elif component_scores.swap_score >= 75:
            return "LIKELY DENY"  # Strong indicators of swap
        
        # ========== STEP 3: ACCESSORIES (if product is same) ==========
        # Accessories missing = fraud
        if component_scores.accessory_score >= 100:
            return "AUTO-DENY"  # Critical accessories missing
        elif component_scores.accessory_score >= 80:
            return "LIKELY DENY"  # Multiple accessories missing
        elif component_scores.accessory_score >= 50:
            return "LIKELY DENY"  # Some accessories missing
        
        # ========== STEP 4: DAMAGE/WEAR/TEAR (only if product verified as same) ==========
        # High intentional damage
        if component_scores.damage_score >= 80:
            return "AUTO-DENY"  # Severe intentional damage
        elif component_scores.damage_score >= 70:
            return "LIKELY DENY"  # Significant damage
        elif component_scores.damage_score >= 50:
            return "MANUAL REVIEW NEEDED"  # Moderate damage - needs review
        
        # High wear (but acceptable for used items)
        if component_scores.wear_score >= 80:
            return "LIKELY DENY"
        
        # ========== DEFAULT SCORING ==========
        if score < 20:
            return "AUTO-APPROVE"
        elif score < 40:
            return "LIKELY APPROVE"
        elif score < 60:
            return "MANUAL REVIEW NEEDED"
        elif score < 80:
            return "LIKELY DENY"
        else:
            return "AUTO-DENY"

    @staticmethod
    def determine_fraud_type(component_scores: ComponentScores) -> Optional[str]:
        """Determine primary fraud type"""
        scores = {
            'Product Swap': component_scores.swap_score,
            'Intentional Damage': component_scores.damage_score,
            'Missing Accessories': component_scores.accessory_score,
            'Used Product Return': component_scores.wear_score,
            'Counterfeit': component_scores.ocr_score
        }

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        if max_score > 50:
            return max_type
        return None


class ReturnFraudDetectionSystem:
    """Main fraud detection system orchestrator"""

    def __init__(self):
        self.normalizer = ImageNormalizer()
        self.ocr_verifier = OCRTextVerification()
        self.accessory_detector = AccessoryDetection()
        self.visual_analyzer = VisualConditionAnalysis()
        self.risk_engine = RiskScoringEngine()

    def process_return(self, delivery_images: List[np.ndarray], 
                      return_images: List[np.ndarray],
                      expected_accessories: Optional[List[str]] = None,
                      product_type: str = None,
                      has_accessories: bool = True) -> FraudAnalysisResult:
        """
        Process return for fraud detection
        
        Args:
            delivery_images: List of delivery images (multiple angles)
            return_images: List of return images (multiple angles)
            expected_accessories: List of expected accessories
            product_type: Type of product (phone, headphones, laptop, etc.)
            has_accessories: Whether product comes with accessories (default: True)
            
        Returns:
            FraudAnalysisResult with fraud score and evidence
            
        Note: Serial number detection is AUTOMATIC via OCR
        """

        # Normalize all images
        delivery_norm = [self.normalizer.normalize_image(img) for img in delivery_images]
        return_norm = [self.normalizer.normalize_image(img) for img in return_images]

        # Use primary angles (front view typically first)
        delivery_primary = delivery_norm[0] if delivery_norm else None
        return_primary = return_norm[0] if return_norm else None

        if delivery_primary is None or return_primary is None:
            raise ValueError("Must provide at least one delivery and one return image")

        # Component 1: OCR Text Verification (Serial Number Check)
        # LOGIC: If serial is in delivery, check if it matches in return
        delivery_text = self.ocr_verifier.extract_text(delivery_primary)
        return_text = self.ocr_verifier.extract_text(return_primary)
        ocr_score, ocr_evidence, serial_in_delivery = self.ocr_verifier.compare_serial_numbers(delivery_text, return_text)

        # Component 2: Accessory Detection (only if accessories shown in delivery)
        # LOGIC: If NO accessories in delivery image, skip check. If accessories present, compare with return.
        accessories_in_delivery = False
        if len(delivery_norm) > 1 and len(return_norm) > 1:
            accessory_score, accessory_evidence, accessories_in_delivery = self.accessory_detector.compare_accessories(
                delivery_norm[1], return_norm[1]
            )
        else:
            accessory_score, accessory_evidence = 0.0, "Accessory images not provided"

        # Component 3: Edge Analysis (Damage Detection)
        damage_score, damage_evidence = self.visual_analyzer.edge_analysis(
            delivery_primary, return_primary
        )

        # Component 4: Keypoint Fingerprinting (Product Swap)
        swap_score, swap_evidence = self.visual_analyzer.keypoint_fingerprinting(
            delivery_primary, return_primary
        )

        # Component 5: Texture & Wear Analysis
        wear_score, wear_evidence = self.visual_analyzer.texture_analysis(
            delivery_primary, return_primary
        )

        # Aggregate component scores
        component_scores = ComponentScores(
            ocr_score=ocr_score,
            accessory_score=accessory_score,
            damage_score=damage_score,
            swap_score=swap_score,
            wear_score=wear_score
        )

        # Calculate final fraud score
        fraud_score = self.risk_engine.calculate_fraud_score(component_scores)
        risk_level = self.risk_engine.determine_risk_level(fraud_score)
        recommendation = self.risk_engine.determine_recommendation(fraud_score, component_scores)
        fraud_type = self.risk_engine.determine_fraud_type(component_scores)

        # Calculate confidence (inverse of component variation)
        component_values = [ocr_score, accessory_score, damage_score, swap_score, wear_score]
        component_std = np.std([s for s in component_values if s > 0]) if any(component_values) else 0
        confidence = max(0.5, 1.0 - (component_std / 100))

        # Compile evidence
        evidence = {
            'ocr': {'score': ocr_score, 'evidence': ocr_evidence},
            'accessory': {'score': accessory_score, 'evidence': accessory_evidence},
            'damage': {'score': damage_score, 'evidence': damage_evidence},
            'swap': {'score': swap_score, 'evidence': swap_evidence},
            'wear': {'score': wear_score, 'evidence': wear_evidence}
        }

        return FraudAnalysisResult(
            fraud_risk_score=fraud_score,
            risk_level=risk_level,
            component_scores=component_scores,
            primary_fraud_type=fraud_type,
            confidence=confidence,
            evidence=evidence,
            timestamp=datetime.now().isoformat(),
            recommendation=recommendation
        )

    def to_dict(self, result: FraudAnalysisResult) -> Dict:
        """Convert result to dictionary for JSON serialization"""
        return {
            'fraud_risk_score': result.fraud_risk_score,
            'risk_level': result.risk_level,
            'component_scores': asdict(result.component_scores),
            'primary_fraud_type': result.primary_fraud_type,
            'confidence': result.confidence,
            'evidence': result.evidence,
            'timestamp': result.timestamp,
            'recommendation': result.recommendation
        }


# Example usage
if __name__ == "__main__":
    print("Fraud Detection System Initialized")
    print("Version 1.0")
    print("\nTo use:")
    print("  1. Load delivery images: cv2.imread('delivery_1.jpg')")
    print("  2. Load return images: cv2.imread('return_1.jpg')")
    print("  3. Initialize system: system = ReturnFraudDetectionSystem()")
    print("  4. Process: result = system.process_return(delivery_imgs, return_imgs)")
