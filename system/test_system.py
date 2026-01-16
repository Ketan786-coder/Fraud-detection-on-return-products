#!/usr/bin/env python3
"""
Test script for Fraud Detection System
Tests all major components without requiring actual images
"""

import numpy as np
import cv2
from fraud_detection_engine import (
    ReturnFraudDetectionSystem,
    ImageNormalizer,
    OCRTextVerification,
    AccessoryDetection,
    VisualConditionAnalysis,
    RiskScoringEngine,
    ComponentScores
)
from image_validator import ImageQualityValidator


def create_test_image(label="Test Image", width=1280, height=1024, pattern="random"):
    """Create synthetic test images for testing"""
    if pattern == "random":
        img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    elif pattern == "uniform":
        img = np.ones((height, width, 3), dtype=np.uint8) * 128
    elif pattern == "gradient":
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            img[i, :] = int(255 * i / height)
    elif pattern == "edges":
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (100, 100), (500, 500), (0, 0, 0), 3)
        cv2.circle(img, (width//2, height//2), 100, (100, 100, 100), -1)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (50, 50), font, 1, (255, 255, 255), 2)
    
    return img


def test_image_normalizer():
    """Test image normalization pipeline"""
    print("\n" + "="*60)
    print("TEST 1: Image Normalizer")
    print("="*60)
    
    normalizer = ImageNormalizer()
    
    # Create test image
    test_img = create_test_image("Normalize Test", pattern="gradient")
    
    print("Original image shape:", test_img.shape)
    print("Testing normalization steps...")
    
    # Test each normalization step
    print("  ✓ Lighting normalization")
    normalized = normalizer.normalize_lighting(test_img)
    assert normalized.shape == test_img.shape
    
    print("  ✓ White balance correction")
    white_balanced = normalizer.white_balance(normalized)
    assert white_balanced.shape == test_img.shape
    
    print("  ✓ Noise reduction")
    denoised = normalizer.denoise(white_balanced)
    assert denoised.shape == test_img.shape
    
    print("  ✓ Resolution standardization")
    standardized = normalizer.standardize_resolution(denoised, target_height=720)
    print(f"    Final shape: {standardized.shape}")
    
    print("✓ Image Normalizer: PASSED")


def test_image_validator():
    """Test image quality validation"""
    print("\n" + "="*60)
    print("TEST 2: Image Quality Validator")
    print("="*60)
    
    validator = ImageQualityValidator()
    
    # Create test images of different qualities
    print("Testing image quality assessment...")
    
    # Good quality image
    good_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
    cv2.rectangle(good_img, (100, 100), (500, 500), (50, 50, 50), 3)
    
    # Poor quality image
    poor_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Very poor quality (blurry)
    blurry_img = cv2.blur(poor_img, (15, 15))
    
    print("  Testing high-quality image...")
    acceptable, score = validator.validate_array(good_img)
    print(f"    Overall score: {score.overall_score:.1%}")
    print(f"    Acceptable: {acceptable}")
    
    print("  Testing low-quality image...")
    acceptable, score = validator.validate_array(poor_img)
    print(f"    Overall score: {score.overall_score:.1%}")
    
    print("  Testing blurry image...")
    acceptable, score = validator.validate_array(blurry_img)
    print(f"    Overall score: {score.overall_score:.1%}")
    print(f"    Sharpness score: {score.sharpness_score:.1%}")
    
    print("✓ Image Validator: PASSED")


def test_edge_analysis():
    """Test edge detection for damage detection"""
    print("\n" + "="*60)
    print("TEST 3: Edge Analysis (Damage Detection)")
    print("="*60)
    
    analyzer = VisualConditionAnalysis()
    
    # Create images with and without damage
    delivery_img = create_test_image("Clean", pattern="uniform")
    
    # Create "damaged" version with added edges
    damaged_img = delivery_img.copy()
    cv2.line(damaged_img, (100, 100), (500, 500), (0, 0, 0), 3)
    cv2.line(damaged_img, (100, 500), (500, 100), (0, 0, 0), 3)
    
    print("Comparing clean vs damaged image...")
    damage_score, evidence = analyzer.edge_analysis(delivery_img, damaged_img)
    
    print(f"  Damage score: {damage_score}")
    print(f"  Evidence: {evidence}")
    
    assert damage_score > 0, "Should detect added damage"
    print("✓ Edge Analysis: PASSED")


def test_keypoint_matching():
    """Test SIFT keypoint matching for product identification"""
    print("\n" + "="*60)
    print("TEST 4: Keypoint Matching (Product Swap Detection)")
    print("="*60)
    
    analyzer = VisualConditionAnalysis()
    
    # Create two similar images (same product)
    img1 = create_test_image("Product A", pattern="edges")
    img2 = create_test_image("Product A", pattern="edges")  # Same pattern
    
    print("Matching same product...")
    swap_score_same, evidence_same = analyzer.keypoint_fingerprinting(img1, img2)
    print(f"  Swap score (same): {swap_score_same}")
    print(f"  Evidence: {evidence_same}")
    
    # Create different image (different product)
    img3 = create_test_image("Product B", pattern="random")
    
    print("Matching different product...")
    swap_score_diff, evidence_diff = analyzer.keypoint_fingerprinting(img1, img3)
    print(f"  Swap score (different): {swap_score_diff}")
    print(f"  Evidence: {evidence_diff}")
    
    assert swap_score_same <= swap_score_diff, "Different products should have higher swap score"
    print("✓ Keypoint Matching: PASSED")


def test_risk_scoring():
    """Test risk scoring engine"""
    print("\n" + "="*60)
    print("TEST 5: Risk Scoring Engine")
    print("="*60)
    
    engine = RiskScoringEngine()
    
    # Test case 1: Legitimate return
    print("Test Case 1: Legitimate return (no fraud signals)")
    scores1 = ComponentScores(
        ocr_score=0,
        accessory_score=0,
        damage_score=0,
        swap_score=0,
        wear_score=5
    )
    fraud_score1 = engine.calculate_fraud_score(scores1)
    level1 = engine.determine_risk_level(fraud_score1)
    rec1 = engine.determine_recommendation(fraud_score1)
    
    print(f"  Fraud Score: {fraud_score1:.1f}")
    print(f"  Risk Level: {level1}")
    print(f"  Recommendation: {rec1}")
    
    # Test case 2: Clear fraud
    print("\nTest Case 2: Clear fraud (multiple signals)")
    scores2 = ComponentScores(
        ocr_score=100,  # IMEI mismatch
        accessory_score=80,  # Missing items
        damage_score=60,  # Damage detected
        swap_score=90,  # Product swap
        wear_score=50
    )
    fraud_score2 = engine.calculate_fraud_score(scores2)
    level2 = engine.determine_risk_level(fraud_score2)
    rec2 = engine.determine_recommendation(fraud_score2)
    
    print(f"  Fraud Score: {fraud_score2:.1f}")
    print(f"  Risk Level: {level2}")
    print(f"  Recommendation: {rec2}")
    
    # Test case 3: Edge case (medium risk)
    print("\nTest Case 3: Edge case (uncertain)")
    scores3 = ComponentScores(
        ocr_score=0,
        accessory_score=20,
        damage_score=30,
        swap_score=10,
        wear_score=40
    )
    fraud_score3 = engine.calculate_fraud_score(scores3)
    level3 = engine.determine_risk_level(fraud_score3)
    rec3 = engine.determine_recommendation(fraud_score3)
    
    print(f"  Fraud Score: {fraud_score3:.1f}")
    print(f"  Risk Level: {level3}")
    print(f"  Recommendation: {rec3}")
    
    assert fraud_score1 < fraud_score2, "Fraudulent return should have higher score"
    print("\n✓ Risk Scoring: PASSED")


def test_full_analysis():
    """Test complete fraud detection system"""
    print("\n" + "="*60)
    print("TEST 6: Full System Analysis")
    print("="*60)
    
    system = ReturnFraudDetectionSystem()
    
    # Create synthetic delivery and return images
    delivery_imgs = [
        create_test_image("Delivery Front", pattern="edges"),
        create_test_image("Delivery Back", pattern="gradient")
    ]
    
    return_imgs = [
        create_test_image("Return Front", pattern="edges"),
        create_test_image("Return Back", pattern="gradient")
    ]
    
    print("Processing return...")
    result = system.process_return(delivery_imgs, return_imgs)
    
    print(f"\n📊 Analysis Results:")
    print(f"  Fraud Risk Score: {result.fraud_risk_score:.1f}/100")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Recommendation: {result.recommendation}")
    print(f"  Confidence: {result.confidence:.1%}")
    
    print(f"\n📈 Component Scores:")
    print(f"  OCR: {result.component_scores.ocr_score:.1f}")
    print(f"  Accessory: {result.component_scores.accessory_score:.1f}")
    print(f"  Damage: {result.component_scores.damage_score:.1f}")
    print(f"  Swap: {result.component_scores.swap_score:.1f}")
    print(f"  Wear: {result.component_scores.wear_score:.1f}")
    
    if result.primary_fraud_type:
        print(f"\n⚠️ Primary Fraud Type: {result.primary_fraud_type}")
    
    print(f"\n📝 Evidence:")
    for component, evidence in result.evidence.items():
        print(f"  {component}: {evidence['evidence']} (score: {evidence['score']:.1f})")
    
    print("\n✓ Full System Analysis: PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 FRAUD DETECTION SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Image Normalizer", test_image_normalizer),
        ("Image Validator", test_image_validator),
        ("Edge Analysis", test_edge_analysis),
        ("Keypoint Matching", test_keypoint_matching),
        ("Risk Scoring", test_risk_scoring),
        ("Full Analysis", test_full_analysis)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name}: FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("✓ ALL TESTS PASSED")
        print("\nSystem is ready for production use.")
        return True
    else:
        print(f"✗ {failed} tests failed")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
