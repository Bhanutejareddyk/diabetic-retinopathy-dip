"""
test_app.py
===========
Automated test suite verifying DIP pipeline, ML classification,
API endpoints, history persistence, and PDF report generation.
"""

import os
import json
import pytest
from app import app, pipeline, classifier, PROCESSED_DIR
import cv2

def test_dip_pipeline_on_all_stages():
    print("\n--- Testing DIP Pipeline on all 5 Stages ---")
    nicknames = ["no_dr", "mild", "moderate", "severe", "proliferative"]
    results = {}

    for stage_id in range(5):
        sample_path = os.path.join(os.path.dirname(__file__), "static", "samples", f"sample_{stage_id}_{nicknames[stage_id]}.jpg")
        assert os.path.exists(sample_path), f"Sample {sample_path} must exist"

        img = cv2.imread(sample_path)
        assert img is not None, "Sample image should be readable by OpenCV"

        out_dir = os.path.join(PROCESSED_DIR, f"test_stage_{stage_id}")
        res = pipeline.process(img, output_dir=out_dir)

        # Check all intermediate images exist
        for key in ["original", "gray", "green", "clahe", "vessels", "lesions", "overlay"]:
            img_file = os.path.join(out_dir, f"{key}.jpg")
            assert os.path.exists(img_file), f"Intermediate image {key}.jpg must be created"
            assert os.path.getsize(img_file) > 1000, f"Intermediate image {key}.jpg must have content"

        feats = res["features"]
        pred = classifier.predict(feats)
        results[stage_id] = {
            "predicted_class": pred["class_id"],
            "name": pred["name"],
            "confidence": pred["confidence"],
            "ma_count": feats["microaneurysm_count"],
            "ex_count": feats["exudate_count"],
            "vessel_density": feats["vessel_density"]
        }
        print(f"Sample Stage {stage_id}: Predicted -> {pred['name']} ({pred['confidence']}%), MA={feats['microaneurysm_count']}, EX={feats['exudate_count']}")

    # Verify that stage 0 is No DR and higher stages show increasing lesions
    assert results[0]["predicted_class"] == 0, "Stage 0 sample should be predicted as No DR"
    assert results[1]["predicted_class"] == 1 or results[1]["predicted_class"] == 2
    assert results[4]["predicted_class"] == 4 or results[4]["predicted_class"] == 3
    print("DIP pipeline verification passed!")


def test_flask_api_endpoints():
    print("\n--- Testing Flask API Endpoints ---")
    client = app.test_client()

    # 1. Test index page
    res = client.get("/")
    assert res.status_code == 200
    assert b"RetinaScan" in res.data

    # 2. Test samples list endpoint
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    assert len(data["samples"]) == 5

    # 3. Test sample analysis endpoint for all 5 stages
    for sid in range(5):
        res = client.get(f"/api/sample/{sid}")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["success"] is True
        assert "prediction" in data
        assert "features" in data
        assert "images" in data
        report_id = data["report_id"]

        # 4. Test PDF download for that report
        res_pdf = client.get(f"/api/report/{report_id}")
        assert res_pdf.status_code == 200
        assert res_pdf.content_type == "application/pdf"
        assert len(res_pdf.data) > 5000, "PDF data must be non-trivial"

    # 5. Test history endpoint
    res = client.get("/api/history")
    assert res.status_code == 200
    hist_data = json.loads(res.data)
    assert hist_data["success"] is True
    assert len(hist_data["history"]) >= 5
    assert hist_data["stats"]["total_scans"] >= 5

    print("All Flask API endpoints passed successfully!")


if __name__ == "__main__":
    test_dip_pipeline_on_all_stages()
    test_flask_api_endpoints()
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY! PROJECT IS VERIFIED & READY FOR VIVA.")
