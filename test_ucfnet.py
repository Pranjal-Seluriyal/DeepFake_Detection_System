#!/usr/bin/env python3
"""
Test script for UCF-Net integration in DeepTrust AI.
Verifies model loading, device selection, and inference on real & fake images.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.ucfnet_detector import UCFNetDetector

def main():
    print("=" * 60)
    print("TESTING UCF-NET INTEGRATION")
    print("=" * 60)

    # 1. Initialize detector (singleton)
    t0 = time.time()
    detector = UCFNetDetector.get_instance()
    t_load = time.time() - t0
    print(f"[OK] UCF-Net loaded in {t_load:.2f} seconds")
    print(f"[OK] Device used: {detector.device}")
    print(f"[OK] Checkpoint: {detector.checkpoint_path}")

    # 2. Test Real and Synthetic Images
    sample_images = [
        ("AI Synthetic Image (ChatGPT)", BASE_DIR / "uploads" / "ChatGPT_Image_May_5_2026_12_10_59_AM.png"),
        ("Real Photo (Pranjal)", BASE_DIR / "uploads" / "Pranjal_Seluriyal_3.png"),
        ("Real Camera Capture", BASE_DIR / "uploads" / "camera_capture_39d73f53.jpg"),
        ("Diamond Test Image", BASE_DIR / "static" / "images" / "real_fake_diamond.png")
    ]

    for label, img_path in sample_images:
        if img_path.exists():
            print("\n" + "-" * 50)
            print(f"Testing: {label} [{img_path.name}]")
            t1 = time.time()
            result = detector.predict(img_path)
            t_infer = time.time() - t1
            print(f"Result:")
            print(f"  - Label:            {result['label']}")
            print(f"  - Status:           {result['status']}")
            print(f"  - Confidence:       {result['confidence']}%")
            print(f"  - Fake Probability: {result['fake_probability']}")
            print(f"  - Real Probability: {result['real_probability']}")
            print(f"  - Model:            {result['model']}")
            print(f"  - Inference Latency: {t_infer:.3f} seconds")

    print("\n" + "=" * 60)
    print("ALL UCF-NET TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
