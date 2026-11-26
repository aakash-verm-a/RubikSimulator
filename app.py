"""
Flask Simulation Server for the 4-Arm Rubik's Cube Solver Robot.

Provides:
  - Interactive 3D 4-arm robot + Rubik's cube simulator UI
  - Webcam photo capture & OpenCV HSV color recognition (from Color_recog-v2.py)
  - Exact Arduino CFOP solver & 4-arm multi-step move decomposition
"""

from __future__ import annotations
import os
import sys
from flask import Flask, jsonify, render_template, request

# Ensure local modules resolve regardless of working directory
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from rubik_solver import Cube, FACE_NAMES, MOVE_DISPLAY
from robot_kinematics import build_simulation_plan, SCAN_SEQUENCE, NEUTRAL_SERVOS
from vision_pipeline import (
    decode_data_url,
    detect_face_from_image,
    visual_grid_to_arduino_string,
    arduino_string_to_visual_grid,
)

app = Flask(
    __name__,
    template_folder=os.path.join(SIM_DIR, "templates"),
    static_folder=os.path.join(SIM_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB (fits Vercel's 4.5 MB limit)


def _cube_to_payload(cube: Cube):
    arduino_dict = cube.to_char_dict()
    visual_dict = {
        face: arduino_string_to_visual_grid(s) for face, s in arduino_dict.items()
    }
    return {"faces_arduino": arduino_dict, "faces_visual": visual_dict}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan_sequence", methods=["GET"])
def api_scan_sequence():
    return jsonify({
        "scan_sequence": SCAN_SEQUENCE,
        "neutral_servos": NEUTRAL_SERVOS,
    })


@app.route("/api/solved_state", methods=["GET"])
def api_solved_state():
    c = Cube()
    return jsonify(_cube_to_payload(c))


@app.route("/api/scramble", methods=["POST"])
def api_scramble():
    data = request.get_json(silent=True) or {}
    count = int(data.get("moves", 18))
    count = max(1, min(40, count))
    c = Cube()
    seq = c.scramble(count)
    payload = _cube_to_payload(c)
    payload["scramble"] = " ".join(MOVE_DISPLAY.get(ch, ch) for ch in seq)
    return jsonify(payload)


@app.route("/api/detect_face", methods=["POST"])
def api_detect_face():
    data = request.get_json(silent=True) or {}
    img_data = data.get("image")
    if not img_data:
        return jsonify({"error": "Missing image payload"}), 400
    try:
        frame = decode_data_url(img_data)
        result = detect_face_from_image(
            frame,
            include_preview=bool(data.get("include_preview", True)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/solve", methods=["POST"])
def api_solve():
    data = request.get_json(silent=True) or {}
    if "faces_arduino" in data:
        faces_arduino = data["faces_arduino"]
    elif "faces_visual" in data:
        faces_arduino = {
            k: visual_grid_to_arduino_string(v)
            for k, v in data["faces_visual"].items()
        }
    else:
        return jsonify({"error": "Missing faces_arduino or faces_visual"}), 400

    try:
        plan = build_simulation_plan(faces_arduino)
        return jsonify(plan)
    except Exception as exc:
        return jsonify({"solved": False, "error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"Starting 4-Arm Rubik's Cube Robot Simulator at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
