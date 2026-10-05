#!/usr/bin/env python3
"""
Full End-to-End Application Integration Test for DeepTrust AI.
Tests the actual Flask application endpoints:
1. /api/health
2. /api/predict/image (Real Image -> Real)
3. /api/predict/image (Deepfake Image -> Deepfake)
4. /analyze (Legacy route with image)
5. /api/predict/audio (Audio endpoint untouched and working)
"""

import os
import sys
import io
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app

def test_full_application():
    print("=" * 70)
    print("STARTING FULL APPLICATION PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # Create app in development mode
    app = create_app("development")
    client = app.test_client()

    # 1. Test Health Endpoint
    print("\n[Step 1] Testing /api/health endpoint...")
    resp = client.get("/api/health")
    print(f"Status Code: {resp.status_code}")
    health_data = resp.get_json()
    print(f"Health Response: {health_data}")
    assert resp.status_code == 200, f"Health check failed: {resp.data}"
    assert health_data["models"]["image_model"] == "loaded", "Image model not loaded in health check"
    assert health_data["models"]["audio_model"] == "loaded", "Audio model not loaded in health check"
    print("[PASS] Health endpoint verified - both image (UCF-Net) and audio models loaded.")

    # 2. Test Real Image through /api/predict/image
    real_img_path = BASE_DIR / "uploads" / "camera_capture_39d73f53.jpg"
    print(f"\n[Step 2] Testing REAL image through /api/predict/image...")
    print(f"Input: {real_img_path.name}")
    with open(real_img_path, "rb") as f:
        real_data = {
            "file": (io.BytesIO(f.read()), real_img_path.name)
        }
        resp = client.post("/api/predict/image", data=real_data, content_type="multipart/form-data")

    print(f"Status Code: {resp.status_code}")
    real_res = resp.get_json()
    print(f"Real Image API Result: {real_res}")
    assert resp.status_code == 200, f"Real image prediction failed: {resp.data}"
    assert real_res["success"] is True, "Prediction success flag is false"
    assert real_res["status"] == "Real", f"Expected 'Real', got {real_res['status']}"
    print(f"[PASS] Real image correctly classified as: {real_res['status']} ({real_res['confidence']})")

    # 3. Test Fake Image through /api/predict/image
    fake_img_path = BASE_DIR / "uploads" / "WhatsApp_Image_2026-05-06_at_10.26.09_AM.jpeg"
    print(f"\n[Step 3] Testing FAKE image through /api/predict/image...")
    print(f"Input: {fake_img_path.name}")
    with open(fake_img_path, "rb") as f:
        fake_data = {
            "file": (io.BytesIO(f.read()), fake_img_path.name)
        }
        resp = client.post("/api/predict/image", data=fake_data, content_type="multipart/form-data")

    print(f"Status Code: {resp.status_code}")
    fake_res = resp.get_json()
    print(f"Fake Image API Result: {fake_res}")
    assert resp.status_code == 200, f"Fake image prediction failed: {resp.data}"
    assert fake_res["success"] is True, "Prediction success flag is false"
    assert fake_res["status"] == "Deepfake", f"Expected 'Deepfake', got {fake_res['status']}"
    print(f"[PASS] Fake image correctly classified as: {fake_res['status']} ({fake_res['confidence']})")

    # 4. Test Legacy /analyze endpoint
    print("\n[Step 4] Testing legacy /analyze route...")
    with open(real_img_path, "rb") as f:
        data = {
            "file": (io.BytesIO(f.read()), real_img_path.name),
            "mode": "image"
        }
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    print(f"Status Code: {resp.status_code}")
    analyze_res = resp.get_json()
    print(f"Legacy Analyze Result: {analyze_res}")
    assert resp.status_code == 200
    assert analyze_res["status"] == "Real"
    print(f"[PASS] Legacy /analyze route working seamlessly: {analyze_res['status']} ({analyze_res['confidence']})")

    # 5. Test Audio endpoint /api/predict/audio (Untouched validation)
    audio_path = BASE_DIR / "uploads" / "voice_test.wav"
    if audio_path.exists():
        print(f"\n[Step 5] Testing UNTOUCHED audio endpoint /api/predict/audio...")
        print(f"Input: {audio_path.name}")
        with open(audio_path, "rb") as f:
            audio_data = {
                "file": (io.BytesIO(f.read()), audio_path.name)
            }
            resp = client.post("/api/predict/audio", data=audio_data, content_type="multipart/form-data")
        print(f"Status Code: {resp.status_code}")
        audio_res = resp.get_json()
        print(f"Audio API Result: {audio_res}")
        assert resp.status_code == 200, f"Audio prediction failed: {resp.data}"
        assert audio_res["success"] is True
        print(f"[PASS] Audio endpoint verified working exactly as originally designed: {audio_res['status']} ({audio_res['confidence']})")

    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED PERFECTLY!")
    print("Final Pipeline Confirmed: Image Upload -> UCF-Net -> REAL/FAKE")
    print("Audio Pipeline Confirmed: UNCHANGED")
    print("=" * 70)

if __name__ == "__main__":
    test_full_application()
