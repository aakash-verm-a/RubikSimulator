"""End-to-end verification tests for the 4-Arm Rubik's Cube Robot Simulator."""

import base64
import os
import sys
import cv2
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from rubik_solver import Cube
from robot_kinematics import build_simulation_plan
import app as sim_app


def run_tests() -> None:
    # 1. Test 200 random scrambles across all 6 starting cube orientations
    ok, phys_ok = 0, 0
    n_trials = 200
    orient_ops = ["", "X", "x", "Y", "y", "Z", "z", "XX"]
    for i in range(n_trials):
        c = Cube()
        c.scramble(25, seed=i)
        for ch in orient_ops[i % len(orient_ops)]:
            c.rubik_rotate(ch)
        plan = build_simulation_plan(c.to_char_dict())
        if plan["solved"]:
            ok += 1
        if plan.get("physical_verified"):
            phys_ok += 1
    print(f"[1] Solver + 4-Arm Physical Replay: {ok}/{n_trials} solved, {phys_ok}/{n_trials} physical-verified")
    assert ok == n_trials and phys_ok == n_trials

    # 2. Test Flask API endpoints
    client = sim_app.app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/scan_sequence").status_code == 200
    assert client.get("/api/solved_state").status_code == 200

    scr = client.post("/api/scramble", json={"moves": 18}).get_json()
    sol = client.post("/api/solve", json={"faces_visual": scr["faces_visual"]}).get_json()
    assert sol["solved"] and sol["physical_verified"]
    print(f"[2] Flask /api/scramble + /api/solve OK: {sol['stats']}")

    # 3. Test automatic full-frame cube detection (off-center + tilted) through /api/detect_face
    bgr_map = {
        "R": (30, 30, 220),
        "O": (20, 130, 245),
        "Y": (30, 230, 240),
        "G": (40, 190, 40),
        "B": (210, 60, 30),
        "W": (240, 240, 240),
    }
    expected = ["R", "O", "Y", "G", "B", "W", "R", "G", "Y"]

    # Build a 180x180 cube face and place it OFF-CENTER in a 640x480 noisy room frame
    cube_patch = np.full((180, 180, 3), 20, dtype=np.uint8)
    for idx, col in enumerate(expected):
        r, c_i = divmod(idx, 3)
        cube_patch[r * 60 + 6 : (r + 1) * 60 - 6, c_i * 60 + 6 : (c_i + 1) * 60 - 6] = bgr_map[col]

    scene = np.full((480, 640, 3), (125, 130, 135), dtype=np.uint8)
    # Place cube off-center at (x=380, y=210) — NOT in the center!
    scene[210:390, 380:560] = cube_patch
    # Also draw a rotated 14-degree version on a second scene
    rot_M = cv2.getRotationMatrix2D((320, 240), 14, 1.0)
    scene_rot = np.full((480, 640, 3), (110, 115, 120), dtype=np.uint8)
    scene_rot[150:330, 230:410] = cube_patch
    scene_rot = cv2.warpAffine(scene_rot, rot_M, (640, 480), borderValue=(110, 115, 120))

    for label, frame in [("off-center", scene), ("tilted-14deg", scene_rot)]:
        _, buf = cv2.imencode(".jpg", frame)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        det = client.post("/api/detect_face", json={"image": data_url, "include_preview": True}).get_json()
        assert det["detected"], f"Cube not detected in {label} scene: {det}"
        assert det["visual_grid"] == expected, f"{label}: Expected {expected}, got {det['visual_grid']}"
        print(f"[3] Auto-Detect ({label}) OK: method={det['method']} grid={det['visual_grid']}")

    # Blank frame should cleanly report detected=False
    blank = np.full((480, 640, 3), 120, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", blank)
    det_blank = client.post(
        "/api/detect_face",
        json={"image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()},
    ).get_json()
    assert det_blank["detected"] is False
    print("[4] Blank frame rejection OK (detected=False)")
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
