"""
app.py
======
Flask backend for Diabetic Retinopathy Detection Using Retinal Images.
Subject: Digital Image Processing (DIP) - Mini / Major Project.
Runs on localhost:5000 and is deployable on Render / Vercel.
"""

import os
import uuid
import json
import datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename

from dip_pipeline import DIPPipeline
from model import classifier, CLASS_INFO
from report_generator import generate_dr_pdf_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("VERCEL"):
    PROCESSED_DIR = "/tmp/processed"
    HISTORY_FILE = "/tmp/history.json"
else:
    PROCESSED_DIR = os.path.join(BASE_DIR, "static", "processed")
    HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB max upload

pipeline = DIPPipeline(target_size=(512, 512))


@app.route("/static/processed/<path:filename>")
def serve_processed_images(filename):
    return send_from_directory(PROCESSED_DIR, filename)


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(records):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")


def is_likely_fundus_image(img_bgr):
    """
    Validation check: Confirms the uploaded image has retinal fundus characteristics:
    - Non-empty, reasonable dimensions
    - Significant reddish/orange hue (R > B usually) or circular field of view
    """
    if img_bgr is None or img_bgr.size == 0:
        return False
    h, w, c = img_bgr.shape
    if h < 64 or w < 64 or c != 3:
        return False

    # Check color channels: in retinal fundus photography, red channel mean is usually higher than blue
    b_mean = float(np.mean(img_bgr[:, :, 0]))
    g_mean = float(np.mean(img_bgr[:, :, 1]))
    r_mean = float(np.mean(img_bgr[:, :, 2]))

    # If completely white or black or inverted
    if (b_mean + g_mean + r_mean) < 15:
        return False

    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_sample_list():
    """
    Returns list of available sample fundus images for each DR stage.
    """
    samples = [
        {"id": 0, "name": "Stage 0 - No DR", "nickname": "no_dr", "color": "#10B981", "desc": "Healthy retina, intact vascular network"},
        {"id": 1, "name": "Stage 1 - Mild", "nickname": "mild", "color": "#EAB308", "desc": "Isolated microaneurysms, minimal exudation"},
        {"id": 2, "name": "Stage 2 - Moderate", "nickname": "moderate", "color": "#F97316", "desc": "Multiple microaneurysms & hard exudates"},
        {"id": 3, "name": "Stage 3 - Severe", "nickname": "severe", "color": "#EF4444", "desc": "Abundant hemorrhages & exudate clusters"},
        {"id": 4, "name": "Stage 4 - Proliferative", "nickname": "proliferative", "color": "#7F1D1D", "desc": "Neovascularization & severe lesion density"}
    ]
    return jsonify({"success": True, "samples": samples})


@app.route("/api/sample/<int:sample_id>", methods=["GET"])
def run_sample_analysis(sample_id):
    """
    Directly runs analysis on one of the 5 canonical sample fundus images.
    """
    sample_id = max(0, min(4, sample_id))
    nicknames = ["no_dr", "mild", "moderate", "severe", "proliferative"]
    sample_filename = f"sample_{sample_id}_{nicknames[sample_id]}.jpg"
    sample_path = os.path.join(SAMPLES_DIR, sample_filename)

    if not os.path.exists(sample_path):
        # Fallback to variant 1
        sample_filename = f"sample_{sample_id}_{nicknames[sample_id]}_1.jpg"
        sample_path = os.path.join(SAMPLES_DIR, sample_filename)

    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": f"Sample image not found on server ({sample_filename}). Run download_dataset.py."}), 404

    img_bgr = cv2.imread(sample_path)
    if img_bgr is None:
        return jsonify({"success": False, "error": "Failed to read sample image"}), 500

    report_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(PROCESSED_DIR, report_id)

    # Run DIP Pipeline
    res = pipeline.process(img_bgr, output_dir=output_dir)
    features = res["features"]

    # Run ML Prediction
    pred = classifier.predict(features)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    images_url = {
        "original_url": f"/static/processed/{report_id}/original.jpg",
        "gray_url": f"/static/processed/{report_id}/gray.jpg",
        "green_url": f"/static/processed/{report_id}/green.jpg",
        "clahe_url": f"/static/processed/{report_id}/clahe.jpg",
        "vessel_url": f"/static/processed/{report_id}/vessels.jpg",
        "optic_disc_url": f"/static/processed/{report_id}/optic_disc.jpg",
        "lesion_url": f"/static/processed/{report_id}/lesions.jpg",
        "overlay_url": f"/static/processed/{report_id}/overlay.jpg"
    }

    record = {
        "report_id": report_id,
        "image_name": sample_filename,
        "source": "Sample",
        "timestamp": now_str,
        "features": features,
        "prediction": pred,
        "images": {
            "original": f"/static/processed/{report_id}/original.jpg",
            "green": f"/static/processed/{report_id}/green.jpg",
            "clahe": f"/static/processed/{report_id}/clahe.jpg",
            "vessels": f"/static/processed/{report_id}/vessels.jpg",
            "optic_disc": f"/static/processed/{report_id}/optic_disc.jpg",
            "lesions": f"/static/processed/{report_id}/lesions.jpg",
            "overlay": f"/static/processed/{report_id}/overlay.jpg"
        }
    }

    history = load_history()
    history.insert(0, record)
    save_history(history[:50])

    return jsonify({
        "success": True,
        "report_id": report_id,
        "prediction": pred,
        "features": features,
        "images": images_url,
        "timestamp": now_str,
        "is_sample": True
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """
    Accepts uploaded fundus image, executes DIP pipeline and ML classifier,
    returns intermediate images and quantitative feature analysis.
    """
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in upload"}), 400

    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    report_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(PROCESSED_DIR, report_id)
    os.makedirs(output_dir, exist_ok=True)

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if not is_likely_fundus_image(img_bgr):
        return jsonify({
            "success": False,
            "error": "Please upload a valid retinal fundus image. The uploaded image does not match retinal optical characteristics."
        }), 400

    # Execute DIP Pipeline
    res = pipeline.process(img_bgr, output_dir=output_dir)
    features = res["features"]

    # Execute ML Classifier
    pred = classifier.predict(features)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    images_url = {
        "original_url": f"/static/processed/{report_id}/original.jpg",
        "gray_url": f"/static/processed/{report_id}/gray.jpg",
        "green_url": f"/static/processed/{report_id}/green.jpg",
        "clahe_url": f"/static/processed/{report_id}/clahe.jpg",
        "vessel_url": f"/static/processed/{report_id}/vessels.jpg",
        "optic_disc_url": f"/static/processed/{report_id}/optic_disc.jpg",
        "lesion_url": f"/static/processed/{report_id}/lesions.jpg",
        "overlay_url": f"/static/processed/{report_id}/overlay.jpg"
    }

    record = {
        "report_id": report_id,
        "image_name": filename,
        "source": "Upload",
        "timestamp": now_str,
        "features": features,
        "prediction": pred,
        "images": {
            "original": f"/static/processed/{report_id}/original.jpg",
            "green": f"/static/processed/{report_id}/green.jpg",
            "clahe": f"/static/processed/{report_id}/clahe.jpg",
            "vessels": f"/static/processed/{report_id}/vessels.jpg",
            "optic_disc": f"/static/processed/{report_id}/optic_disc.jpg",
            "lesions": f"/static/processed/{report_id}/lesions.jpg",
            "overlay": f"/static/processed/{report_id}/overlay.jpg"
        }
    }

    history = load_history()
    history.insert(0, record)
    save_history(history[:50])

    return jsonify({
        "success": True,
        "report_id": report_id,
        "prediction": pred,
        "features": features,
        "images": images_url,
        "timestamp": now_str,
        "is_sample": False
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Returns scan history plus aggregate stats (Total scans, Normal vs DR detected).
    """
    history = load_history()
    total = len(history)
    normal_count = sum(1 for h in history if h.get("prediction", {}).get("class_id") == 0)
    dr_count = total - normal_count

    stats = {
        "total_scans": total,
        "normal_scans": normal_count,
        "dr_detected": dr_count,
        "normal_pct": round((normal_count / total * 100), 1) if total > 0 else 0.0,
        "dr_pct": round((dr_count / total * 100), 1) if total > 0 else 0.0,
        "class_breakdown": {
            "0": sum(1 for h in history if h.get("prediction", {}).get("class_id") == 0),
            "1": sum(1 for h in history if h.get("prediction", {}).get("class_id") == 1),
            "2": sum(1 for h in history if h.get("prediction", {}).get("class_id") == 2),
            "3": sum(1 for h in history if h.get("prediction", {}).get("class_id") == 3),
            "4": sum(1 for h in history if h.get("prediction", {}).get("class_id") == 4)
        }
    }

    return jsonify({
        "success": True,
        "history": history,
        "stats": stats
    })


@app.route("/api/history/<report_id>", methods=["DELETE"])
def delete_history_item(report_id):
    history = load_history()
    filtered = [h for h in history if h.get("report_id") != report_id]
    save_history(filtered)
    return jsonify({"success": True, "message": "Record deleted"})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    save_history([])
    return jsonify({"success": True, "message": "History cleared"})


@app.route("/api/report/<report_id>", methods=["GET"])
def download_pdf_report(report_id):
    """
    Builds and downloads PDF Diagnostic Report for the specified report_id.
    """
    history = load_history()
    matched = next((h for h in history if h.get("report_id") == report_id), None)

    if not matched:
        return jsonify({"success": False, "error": "Report record not found"}), 404

    output_dir = os.path.join(PROCESSED_DIR, report_id)
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"DR_Report_{report_id}.pdf")

    matched_with_dir = matched.copy()
    matched_with_dir["processed_dir"] = output_dir

    try:
        generate_dr_pdf_report(matched_with_dir, pdf_path)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"Diabetic_Retinopathy_Report_{report_id}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to generate PDF: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("DIABETIC RETINOPATHY DETECTION - FLASK DIP WEB SERVER")
    print("=" * 60)
    print("Running on http://localhost:5000")
    print("Press Ctrl+C to terminate.")
    app.run(host="0.0.0.0", port=5000, debug=True)
