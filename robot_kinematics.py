"""
4-Arm Rubik's Cube Robot Kinematics & Servo Sequence Model.

Faithfully models the physical 8-servo, 4-arm robot from:
  - servo_and_motor_rotations.ino
  - Rubik.ino (Rubik_Rotate motor mappings)
  - rubik_orient_SolveCross.ino (self_orient motor mappings)
  - store_and_print.ino (sequencia_de_armazenamento scan sequence)

Physical geometry:
  - 4 arms surround the cube horizontally: FRONT, BACK, LEFT, RIGHT.
  - Each arm has 2 servos:
      * Down_<Arm>: linear gripper (clamped ~122-137 deg, retracted ~60 deg, open ~10-30 deg)
      * Up_<Arm>:   wrist rotator  (home ~80-83 deg, CW ~170-180 deg, CCW ~0-8 deg)
  - Logical cube face -> Physical arm mapping (from Rubik.ino):
      * R / R' -> Right arm (1 macro step)
      * L / L' -> Left arm  (1 macro step)
      * U / U' -> Front arm (1 macro step)
      * D / D' -> Back arm  (1 macro step)
      * F / F' -> Multi-step (3 macro steps): Movex -> Back arm -> MoveX
      * B / B' -> Multi-step (3 macro steps): Movex -> Front arm -> MoveX
      * Y / Y' -> Whole-cube rotation using Front+Back arms (1 macro step)
      * X / X' -> Whole-cube rotation using Left+Right arms (1 macro step)
      * Z / Z' -> Multi-step (3 macro steps): MoveX -> MoveY/Movey -> Movex
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

from rubik_solver import Cube, SolveResult, MOVE_DISPLAY

NEUTRAL_SERVOS: Dict[str, int] = {
    "Up_Right": 82,
    "Up_Back": 80,
    "Up_Left": 83,
    "Up_Front": 80,
    "Down_Right": 130,
    "Down_Back": 130,
    "Down_Left": 130,
    "Down_Front": 130,
}


def _merge_servos(base: Dict[str, int], updates: Dict[str, int]) -> Dict[str, int]:
    out = dict(base)
    out.update(updates)
    return out


def _face_turn_keyframes(
    arm: str,
    target_up: int,
    home_up: int,
    clamp_others: Tuple[str, str],
    down_hold: int = 123,
    other_clamp: int = 135,
    d1: int = 200,
    d2: int = 350,
    d3: int = 200,
    d4: int = 200,
    d5: int = 200,
) -> List[Dict[str, Any]]:
    s = dict(NEUTRAL_SERVOS)
    s1 = _merge_servos(s, {f"Down_{clamp_others[0]}": other_clamp, f"Down_{clamp_others[1]}": other_clamp, f"Down_{arm}": down_hold})
    s2 = _merge_servos(s1, {f"Up_{arm}": target_up})
    s3 = _merge_servos(s2, {f"Down_{arm}": 60})
    s4 = _merge_servos(s3, {f"Up_{arm}": home_up})
    s5 = _merge_servos(s4, {"Down_Right": 125, "Down_Back": 130, "Down_Left": 125, "Down_Front": 130})
    return [
        {"phase": f"Clamp cube & engage {arm} arm", "duration_ms": d1, "servos": s1, "anim": "clamp"},
        {"phase": f"Rotate {arm} wrist to {target_up}°", "duration_ms": d2, "servos": s2, "anim": "rotate_face"},
        {"phase": f"Retract {arm} arm (release)", "duration_ms": d3, "servos": s3, "anim": "retract"},
        {"phase": f"Return {arm} wrist to home ({home_up}°)", "duration_ms": d4, "servos": s4, "anim": "reset_wrist"},
        {"phase": "Re-grip all 4 arms", "duration_ms": d5, "servos": s5, "anim": "regrip"},
    ]


def _cube_rot_x_keyframes(up_right: int, up_left: int, direction_label: str) -> List[Dict[str, Any]]:
    s = dict(NEUTRAL_SERVOS)
    s1 = _merge_servos(s, {"Down_Back": 60, "Down_Front": 60, "Down_Left": 132, "Down_Right": 132})
    s2 = _merge_servos(s1, {"Up_Right": up_right, "Up_Left": up_left})
    s3 = _merge_servos(s2, {"Down_Back": 130, "Down_Front": 130})
    s4 = _merge_servos(s3, {"Down_Right": 60, "Down_Left": 60})
    s5 = _merge_servos(s4, {"Up_Right": 82, "Up_Left": 83})
    s6 = _merge_servos(s5, {"Down_Right": 125, "Down_Back": 130, "Down_Left": 125, "Down_Front": 130})
    return [
        {"phase": "Retract Front & Back arms; grip Left & Right", "duration_ms": 200, "servos": s1, "anim": "release_pair"},
        {"phase": f"Rotate whole cube {direction_label} via Left+Right arms", "duration_ms": 200, "servos": s2, "anim": "rotate_cube"},
        {"phase": "Re-grip Front & Back arms", "duration_ms": 200, "servos": s3, "anim": "regrip_pair"},
        {"phase": "Retract Left & Right arms", "duration_ms": 150, "servos": s4, "anim": "retract_pair"},
        {"phase": "Reset Left & Right wrists to home", "duration_ms": 200, "servos": s5, "anim": "reset_wrist"},
        {"phase": "Re-grip all 4 arms", "duration_ms": 150, "servos": s6, "anim": "regrip"},
    ]


def _cube_rot_y_keyframes(up_back: int, up_front: int, direction_label: str) -> List[Dict[str, Any]]:
    s = dict(NEUTRAL_SERVOS)
    s1 = _merge_servos(s, {"Down_Back": 137, "Down_Front": 137, "Down_Left": 60, "Down_Right": 60})
    s2 = _merge_servos(s1, {"Up_Back": up_back, "Up_Front": up_front})
    s3 = _merge_servos(s2, {"Down_Left": 125, "Down_Right": 125})
    s4 = _merge_servos(s3, {"Down_Back": 60, "Down_Front": 60})
    s5 = _merge_servos(s4, {"Up_Back": 80, "Up_Front": 80})
    s6 = _merge_servos(s5, {"Down_Right": 125, "Down_Back": 130, "Down_Left": 125, "Down_Front": 130})
    return [
        {"phase": "Retract Left & Right arms; grip Front & Back", "duration_ms": 300, "servos": s1, "anim": "release_pair"},
        {"phase": f"Rotate whole cube {direction_label} via Front+Back arms", "duration_ms": 300, "servos": s2, "anim": "rotate_cube"},
        {"phase": "Re-grip Left & Right arms", "duration_ms": 320, "servos": s3, "anim": "regrip_pair"},
        {"phase": "Retract Front & Back arms", "duration_ms": 200, "servos": s4, "anim": "retract_pair"},
        {"phase": "Reset Front & Back wrists to home", "duration_ms": 200, "servos": s5, "anim": "reset_wrist"},
        {"phase": "Re-grip all 4 arms", "duration_ms": 150, "servos": s6, "anim": "regrip"},
    ]


# Exact catalog of physical servo primitives from servo_and_motor_rotations.ino
SERVO_PRIMITIVES: Dict[str, Dict[str, Any]] = {
    "ServosFace_RightCW": {
        "fn": "ServosFace_RightCW",
        "label": "Right Arm CW (90°)",
        "kind": "face_turn",
        "arms": ["RIGHT"],
        "physical_face": "RIGHT",
        "dir": 1,
        "keyframes": _face_turn_keyframes("Right", 180, 82, ("Front", "Back"), 123, 137),
    },
    "ServosFace_RightCCW": {
        "fn": "ServosFace_RightCCW",
        "label": "Right Arm CCW (-90°)",
        "kind": "face_turn",
        "arms": ["RIGHT"],
        "physical_face": "RIGHT",
        "dir": -1,
        "keyframes": _face_turn_keyframes("Right", 0, 82, ("Front", "Back"), 123, 137),
    },
    "ServosFace_LeftCW": {
        "fn": "ServosFace_LeftCW",
        "label": "Left Arm CW (90°)",
        "kind": "face_turn",
        "arms": ["LEFT"],
        "physical_face": "LEFT",
        "dir": 1,
        "keyframes": _face_turn_keyframes("Left", 180, 83, ("Front", "Back"), 123, 137, d1=150, d2=350, d3=150, d4=200, d5=200),
    },
    "ServosFace_LeftCCW": {
        "fn": "ServosFace_LeftCCW",
        "label": "Left Arm CCW (-90°)",
        "kind": "face_turn",
        "arms": ["LEFT"],
        "physical_face": "LEFT",
        "dir": -1,
        "keyframes": _face_turn_keyframes("Left", 0, 83, ("Front", "Back"), 122, 137),
    },
    "ServosFace_FrontCW": {
        "fn": "ServosFace_FrontCW",
        "label": "Front Arm CW (90°)",
        "kind": "face_turn",
        "arms": ["FRONT"],
        "physical_face": "UP",
        "dir": 1,
        "keyframes": _face_turn_keyframes("Front", 180, 80, ("Left", "Right"), 123, 132),
    },
    "ServosFace_FrontCCW": {
        "fn": "ServosFace_FrontCCW",
        "label": "Front Arm CCW (-90°)",
        "kind": "face_turn",
        "arms": ["FRONT"],
        "physical_face": "UP",
        "dir": -1,
        "keyframes": _face_turn_keyframes("Front", 1, 80, ("Left", "Right"), 123, 132),
    },
    "ServosFace_BackCW": {
        "fn": "ServosFace_BackCW",
        "label": "Back Arm CW (90°)",
        "kind": "face_turn",
        "arms": ["BACK"],
        "physical_face": "DOWN",
        "dir": 1,
        "keyframes": _face_turn_keyframes("Back", 180, 80, ("Left", "Right"), 123, 132),
    },
    "ServosFace_BackCCW": {
        "fn": "ServosFace_BackCCW",
        "label": "Back Arm CCW (-90°)",
        "kind": "face_turn",
        "arms": ["BACK"],
        "physical_face": "DOWN",
        "dir": -1,
        "keyframes": _face_turn_keyframes("Back", 0, 80, ("Left", "Right"), 122, 132),
    },
    "ServosCube_MoveX": {
        "fn": "ServosCube_MoveX",
        "label": "Whole Cube X (+90°)",
        "kind": "cube_rotate",
        "arms": ["LEFT", "RIGHT"],
        "axis": "X",
        "dir": 1,
        "keyframes": _cube_rot_x_keyframes(173, 4, "X (+90°)"),
    },
    "ServosCube_Movex": {
        "fn": "ServosCube_Movex",
        "label": "Whole Cube X' (-90°)",
        "kind": "cube_rotate",
        "arms": ["LEFT", "RIGHT"],
        "axis": "X",
        "dir": -1,
        "keyframes": _cube_rot_x_keyframes(5, 174, "X' (-90°)"),
    },
    "ServosCube_MoveY": {
        "fn": "ServosCube_MoveY",
        "label": "Whole Cube Y (+90°)",
        "kind": "cube_rotate",
        "arms": ["FRONT", "BACK"],
        "axis": "Y",
        "dir": 1,
        "keyframes": _cube_rot_y_keyframes(170, 8, "Y (+90°)"),
    },
    "ServosCube_Movey": {
        "fn": "ServosCube_Movey",
        "label": "Whole Cube Y' (-90°)",
        "kind": "cube_rotate",
        "arms": ["FRONT", "BACK"],
        "axis": "Y",
        "dir": -1,
        "keyframes": _cube_rot_y_keyframes(5, 171, "Y' (-90°)"),
    },
}

# Precompute duration_ms for each primitive
for _k, _v in SERVO_PRIMITIVES.items():
    _v["duration_ms"] = sum(kf["duration_ms"] for kf in _v["keyframes"])


# Exact logical move -> physical servo macro sequence (from Rubik.ino & servo_and_motor_rotations.ino)
LOGICAL_TO_MACROS: Dict[str, List[str]] = {
    "R": ["ServosFace_RightCW"],
    "r": ["ServosFace_RightCCW"],
    "L": ["ServosFace_LeftCW"],
    "l": ["ServosFace_LeftCCW"],
    "U": ["ServosFace_FrontCW"],
    "u": ["ServosFace_FrontCCW"],
    "D": ["ServosFace_BackCW"],
    "d": ["ServosFace_BackCCW"],
    # Multi-step moves (no physical arm directly on logical Front or Back)
    "F": ["ServosCube_Movex", "ServosFace_BackCW", "ServosCube_MoveX"],
    "f": ["ServosCube_Movex", "ServosFace_BackCCW", "ServosCube_MoveX"],
    "B": ["ServosCube_Movex", "ServosFace_FrontCW", "ServosCube_MoveX"],
    "b": ["ServosCube_Movex", "ServosFace_FrontCCW", "ServosCube_MoveX"],
    # Whole-cube rotations
    "X": ["ServosCube_MoveX"],
    "x": ["ServosCube_Movex"],
    "Y": ["ServosCube_MoveY"],
    "y": ["ServosCube_Movey"],
    "Z": ["ServosCube_MoveX", "ServosCube_MoveY", "ServosCube_Movex"],
    "z": ["ServosCube_MoveX", "ServosCube_Movey", "ServosCube_Movex"],
}

MOVE_EXPLANATIONS: Dict[str, str] = {
    "R": "Direct 1-step turn using the Right arm (CW).",
    "r": "Direct 1-step turn using the Right arm (CCW).",
    "L": "Direct 1-step turn using the Left arm (CW).",
    "l": "Direct 1-step turn using the Left arm (CCW).",
    "U": "Direct 1-step turn using the Front arm (CW) — grips logical Up face.",
    "u": "Direct 1-step turn using the Front arm (CCW) — grips logical Up face.",
    "D": "Direct 1-step turn using the Back arm (CW) — grips logical Down face.",
    "d": "Direct 1-step turn using the Back arm (CCW) — grips logical Down face.",
    "F": "3-step composite move: no arm on Front face -> tilt cube (X'), turn Back arm CW, tilt back (X).",
    "f": "3-step composite move: no arm on Front face -> tilt cube (X'), turn Back arm CCW, tilt back (X).",
    "B": "3-step composite move: no arm on Back face -> tilt cube (X'), turn Front arm CW, tilt back (X).",
    "b": "3-step composite move: no arm on Back face -> tilt cube (X'), turn Front arm CCW, tilt back (X).",
    "X": "Whole-cube rotation (+90° around X) using Left & Right arms.",
    "x": "Whole-cube rotation (-90° around X) using Left & Right arms.",
    "Y": "Whole-cube rotation (+90° around Y) using Front & Back arms.",
    "y": "Whole-cube rotation (-90° around Y) using Front & Back arms.",
    "Z": "3-step whole-cube Z rotation: MoveX -> MoveY -> Movex.",
    "z": "3-step whole-cube Z' rotation: MoveX -> Movey -> Movex.",
}

# Exact 6-face camera scanning sequence from store_and_print.ino (sequencia_de_armazenamento)
SCAN_SEQUENCE = [
    {
        "step": 1,
        "face": "RIGHT",
        "robot_action": "clearVision() — Retract Left & Right arms, hold with Front+Back, present RIGHT face to camera.",
        "instruction": "Show the RIGHT face (typically Green center) to the webcam with UP on top.",
    },
    {
        "step": 2,
        "face": "LEFT",
        "robot_action": "Rotate Front+Back arms 180° (Up_Back->170°, Up_Front->8°) to present LEFT face to camera.",
        "instruction": "Flip 180° horizontally to show the LEFT face (typically Blue center).",
    },
    {
        "step": 3,
        "face": "UP",
        "robot_action": "Transfer grip to Left+Right arms, retract Front+Back, tilt 90° to present UP face to camera.",
        "instruction": "Show the UP face (typically Yellow center) with BACK at top and FRONT at bottom.",
    },
    {
        "step": 4,
        "face": "DOWN",
        "robot_action": "Rotate Left+Right arms 180° (Up_Right->173°, Up_Left->4°) to present DOWN face to camera.",
        "instruction": "Show the DOWN face (typically White center) with FRONT at top and BACK at bottom.",
    },
    {
        "step": 5,
        "face": "FRONT",
        "robot_action": "ServosCube_MoveY() + clearVision() — Re-grip and present FRONT face to camera.",
        "instruction": "Show the FRONT face (typically Red center) with UP on top.",
    },
    {
        "step": 6,
        "face": "BACK",
        "robot_action": "Rotate Front+Back arms 180° to present BACK face, then ServosCube_Movey() back to home.",
        "instruction": "Show the BACK face (typically Orange center) with UP on top.",
    },
]

# Inter-move pause from Rubik.ino line 235: delay(700)
INTER_MOVE_DELAY_MS = 700


def _macro_to_cube_op(macro_fn: str) -> str:
    """Maps an individual physical macro step to the logical Cube rotation it performs."""
    return {
        "ServosFace_RightCW": "R",
        "ServosFace_RightCCW": "r",
        "ServosFace_LeftCW": "L",
        "ServosFace_LeftCCW": "l",
        "ServosFace_FrontCW": "U",
        "ServosFace_FrontCCW": "u",
        "ServosFace_BackCW": "D",
        "ServosFace_BackCCW": "d",
        "ServosCube_MoveX": "X",
        "ServosCube_Movex": "x",
        "ServosCube_MoveY": "Y",
        "ServosCube_Movey": "y",
    }[macro_fn]


def build_simulation_plan(initial_faces_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    Runs the exact Arduino solver on `initial_faces_dict` and builds the full
    step-by-step physical robot simulation plan including:
      - Logical moves & stage tags
      - Multi-step 4-arm macro decomposition
      - Per-substep cube state snapshots (so stepping through sub-steps animates accurately!)
      - Exact servo angle keyframes & real-world timing estimates
    """
    cube = Cube()
    cube.set_from_char_faces(initial_faces_dict)
    result: SolveResult = cube.solve()

    if not result.solved:
        return {
            "solved": False,
            "error": result.error or "Failed to solve cube.",
            "initial_state": result.initial_state,
        }

    # Replay from initial state to capture exact cube state after every logical move AND every physical macro step
    sim_cube = Cube()
    sim_cube.set_from_char_faces(initial_faces_dict)

    orient_len = len(result.orient_moves)
    timeline: List[Dict[str, Any]] = []
    total_macro_steps = 0
    multi_step_count = 0
    total_duration_ms = 0

    for idx, mv in enumerate(result.full_moves):
        stage = "Orient" if idx < orient_len else "Solve (CFOP)"
        macro_names = LOGICAL_TO_MACROS[mv]
        is_multi = len(macro_names) > 1
        if is_multi:
            multi_step_count += 1

        state_before = sim_cube.to_char_matrix()
        macro_entries = []
        move_duration_ms = 0

        for sub_idx, fn_name in enumerate(macro_names):
            prim = SERVO_PRIMITIVES[fn_name]
            sub_before = sim_cube.to_char_matrix()
            # Apply the physical macro step directly to sim_cube!
            sim_cube.rubik_rotate(_macro_to_cube_op(fn_name))
            sub_after = sim_cube.to_char_matrix()

            macro_entries.append({
                "sub_index": sub_idx,
                "fn": fn_name,
                "label": prim["label"],
                "kind": prim["kind"],
                "arms": prim["arms"],
                "physical_face": prim.get("physical_face"),
                "axis": prim.get("axis"),
                "dir": prim["dir"],
                "duration_ms": prim["duration_ms"],
                "keyframes": prim["keyframes"],
                "state_before": sub_before,
                "state_after": sub_after,
            })
            move_duration_ms += prim["duration_ms"]
            total_macro_steps += 1

        state_after = sim_cube.to_char_matrix()
        total_duration_ms += move_duration_ms + INTER_MOVE_DELAY_MS

        timeline.append({
            "index": idx,
            "move_char": mv,
            "notation": MOVE_DISPLAY.get(mv, mv),
            "stage": stage,
            "is_multi_step": is_multi,
            "explanation": MOVE_EXPLANATIONS.get(mv, ""),
            "duration_ms": move_duration_ms,
            "macro_steps": macro_entries,
            "state_before": state_before,
            "state_after": state_after,
        })

    # Verify that replaying physical macro steps ALSO ends in the solved state!
    physical_verified = sim_cube.is_solved()

    return {
        "solved": True,
        "physical_verified": physical_verified,
        "initial_state": result.initial_state,
        "final_state": sim_cube.to_char_matrix(),
        "orient_moves": [MOVE_DISPLAY[c] for c in result.orient_moves],
        "optimized_moves": [MOVE_DISPLAY[c] for c in result.optimized_solve_moves],
        "full_moves_notation": [MOVE_DISPLAY[c] for c in result.full_moves],
        "stage_breakdown": [
            {
                "stage": s.stage,
                "raw_move_count": len(s.raw_moves),
                "moves": [MOVE_DISPLAY.get(c, c) for c in s.raw_moves],
            }
            for s in result.stage_breakdown
        ],
        "stats": {
            "logical_moves": len(result.full_moves),
            "raw_cfop_moves": len(result.raw_solve_moves),
            "optimized_cfop_moves": len(result.optimized_solve_moves),
            "moves_saved_by_optimizer": len(result.raw_solve_moves) - len(result.optimized_solve_moves),
            "physical_macro_steps": total_macro_steps,
            "multi_step_moves": multi_step_count,
            "estimated_physical_seconds": round(total_duration_ms / 1000.0, 1),
        },
        "timeline": timeline,
    }
