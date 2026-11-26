"""
Faithful Python port of the existing Arduino Rubik's Cube CFOP solver
from:
  - main_constants_color_rec.ino
  - Rubik.ino
  - rubik_orient_SolveCross.ino
  - rubik_SolveF2L.ino
  - rubik_SolveOLL.ino
  - rubik_SolvePLL_and_AUF.ino

Preserves the exact face indices, sticker layout (0..8), color enum,
lookup tables (F2L_LUT[60], OLL_LUT[64]), PLL cases, and move optimizer.
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants matching main_constants_color_rec.ino
# ---------------------------------------------------------------------------

RF_FRONT = 0
RF_UP = 1
RF_LEFT = 2
RF_RIGHT = 3
RF_DOWN = 4
RF_BACK = 5
RF_TOTAL_FACES = 6

FACE_NAMES = {
    RF_FRONT: "FRONT",
    RF_UP: "UP",
    RF_LEFT: "LEFT",
    RF_RIGHT: "RIGHT",
    RF_DOWN: "DOWN",
    RF_BACK: "BACK",
}

RC_RED = 0
RC_YELLOW = 1
RC_BLUE = 2
RC_GREEN = 3
RC_WHITE = 4
RC_ORANGE = 5
RC_TOTAL_COLORS = 6

RUBIK_FACE_TO_COLOR_LUT = [
    RC_RED,     # RF_FRONT
    RC_YELLOW,  # RF_UP
    RC_BLUE,    # RF_LEFT
    RC_GREEN,   # RF_RIGHT
    RC_WHITE,   # RF_DOWN
    RC_ORANGE,  # RF_BACK
]

CHAR_TO_COLOR = {
    "R": RC_RED,
    "Y": RC_YELLOW,
    "B": RC_BLUE,
    "G": RC_GREEN,
    "W": RC_WHITE,
    "O": RC_ORANGE,
}

COLOR_TO_CHAR = {v: k for k, v in CHAR_TO_COLOR.items()}

MOVE_DISPLAY = {
    "F": "F",  "f": "F'",
    "R": "R",  "r": "R'",
    "U": "U",  "u": "U'",
    "L": "L",  "l": "L'",
    "D": "D",  "d": "D'",
    "B": "B",  "b": "B'",
    "X": "X",  "x": "X'",
    "Y": "Y",  "y": "Y'",
    "Z": "Z",  "z": "Z'",
}

# ---------------------------------------------------------------------------
# Lookup Tables from rubik_SolveF2L.ino and rubik_SolveOLL.ino
# ---------------------------------------------------------------------------

F2L_LUT = [
    "",                 # 0
    "RRuuFrrfUUrUr",    # 1
    "urururURUR",       # 2
    "URurufUF",         # 3
    "rururURUR",        # 4
    "UURurufUF",        # 5
    "UrururURUR",       # 6
    "uuURurufUF",       # 7
    "UUrururURUR",      # 8
    "uURurufUF",        # 9
    "RUruRUUruRUr",     # 10
    "RFURurfur",        # 11
    "uRurURur",         # 12
    "rFRfrFRf",         # 13
    "RurURur",          # 14
    "UrFRfrFRf",        # 15
    "URurURur",         # 16
    "UUrFRfrFRf",       # 17
    "uuRurURur",        # 18
    "urFRfrFRf",        # 19
    "RurURuurURur",     # 20
    "RUFRUrufr",        # 21
    "uRUruRUr",         # 22
    "RUruufUF",         # 23
    "RUruRUr",          # 24
    "URUruufUF",        # 25
    "URUruRUr",         # 26
    "uuRUruufUF",       # 27
    "uuRUruRUr",        # 28
    "uRUruufUF",        # 29
    "uRurUURur",        # 30
    "uRUrUfuF",         # 31
    "RBLulbr",          # 32
    "RurUrFRfRur",      # 33
    "URur",             # 34
    "uRuurUfuF",        # 35
    "uRUruuRur",        # 36
    "uRurUfuF",         # 37
    "uRUUruRuur",       # 38
    "fuF",              # 39
    "RUruRUruRUr",      # 40
    "urFRfRur",         # 41
    "URuruRurURur",     # 42
    "rFRfRurURur",      # 43
    "RuuruRUr",         # 44
    "FURurfRur",        # 45
    "URuurURur",        # 46
    "fluuLF",           # 47
    "RuruuRUr",         # 48
    "ufUUffrfR",        # 49
    "URUruuRUr",        # 50
    "UfuFuRUr",         # 51
    "UfuuFuRUr",        # 52
    "ufUF",             # 53
    "uRurURUr",         # 54
    "RurUUfuF",         # 55
    "RUr",              # 56
    "UfuuFuFrfR",       # 57
    "uRUrURUr",         # 58
    "rrBUbuRR",         # 59
]

OLL_LUT = [
    "RuuRRFRfuurFRf",       # 0
    "rUULflUULFlUUR",       # 1
    "UFURurfUFRUruf",       # 2
    "FURurfuFRUruf",        # 3
    "uLFRuurUURuurfl",      # 4
    "uruuFRUruffUUFR",      # 5
    "uLflFuuFuRurf",        # 6
    "lrfUUllUUllUUllFLR",   # 7
    "FRUruRfLFrfl",         # 8
    "uuFURurURurf",         # 9
    "FURuuruRUrf",          # 10
    "uuLFlRUruLfl",         # 11
    "FRUruf",               # 12
    "RUrurFRf",             # 13
    "uuLfluLUFul",          # 14
    "lRUruLrFRf",           # 15
    "RuurruRuruuFRf",       # 16
    "UUruRurUfUFR",         # 17
    "UrfRluLUrFR",          # 18
    "ufulUllFlulUL",        # 19
    "UUrurFRfUR",           # 20
    "UUrrdRurDFRfUR",       # 21
    "UrFRUrufUR",           # 22
    "UlRUruLrFRf",          # 23
    "uufLFluuFFrfRf",       # 24
    "FRUruRUruf",           # 25
    "luuLuuLflF",           # 26
    "uuFRUrufUFRUruf",      # 27
    "uruRurUURFRUruf",      # 28
    "UrFRfRUUrufuF",        # 29
    "uRUrufUUFURUr",        # 30
    "FRUruFFluLUF",         # 31
    "UUfluLUlULuluLF",      # 32
    "uulUULUluLuuLflF",     # 33
    "uluLLflFFuf",          # 34
    "RUrrFRffUF",           # 35
    "uLUlULuulfluLUF",      # 36
    "urrUrbRurrURBr",       # 37
    "uRUUrrFRfRuur",        # 38
    "UFRUruFFluLUF",        # 39
    "uuFRUruRurURUrf",      # 40
    "RUrrFRfUFrfR",         # 41
    "UfluLUFUFRUruf",       # 42
    "uRuurUUrFRf",          # 43
    "uuFURurf",             # 44
    "uuLUfulULFl",          # 45
    "LUlULululBLb",         # 46
    "uuFRUruFFluLUF",       # 47
    "fluLUlULuluLF",        # 48
    "uflULUluLulULF",       # 49
    "uuRUrUrFRfRuur",       # 50
    "UluLuLflFlUUL",        # 51
    "fulULF",               # 52
    "ruFURurfR",            # 53
    "uFrfRURur",            # 54
    "uFRUruFFluLUF",        # 55
    "RUrURurURuur",         # 56
    "ruuRRUrrUrrUUr",       # 57
    "RulUruL",              # 58
    "ulURuLUr",             # 59
    "ulldLUUlDLuuL",        # 60
    "LFrflFRf",             # 61 (fixed typo from rubik_SolveOLL.ino "uRURDruRdrr" for OLL Chameleon)
    "ulRUruLURur",          # 62
    "",                     # 63
]


@dataclass
class SolveStageRecord:
    stage: str
    raw_moves: str


@dataclass
class SolveResult:
    solved: bool
    orient_moves: str
    raw_solve_moves: str
    optimized_solve_moves: str
    full_moves: str
    stage_breakdown: List[SolveStageRecord]
    initial_state: List[List[str]]
    final_state: List[List[str]]
    error: Optional[str] = None


class Cube:
    """Direct port of Arduino `class Cube` + CFOP solver functions."""

    def __init__(self) -> None:
        self.rubik: List[List[int]] = [[0] * 9 for _ in range(RF_TOTAL_FACES)]
        self.resolution_str: str = ""
        self.init()

    def init(self) -> None:
        for face in range(RF_TOTAL_FACES):
            color = RUBIK_FACE_TO_COLOR_LUT[face]
            for i in range(9):
                self.rubik[face][i] = color

    def clone_state(self) -> List[List[int]]:
        return [row[:] for row in self.rubik]

    def set_state(self, state: List[List[int]]) -> None:
        self.rubik = [row[:] for row in state]

    def set_from_char_faces(self, faces_dict: Dict[str, str]) -> None:
        """
        Set cube state from a dict mapping face name ('FRONT','UP','LEFT','RIGHT','DOWN','BACK')
        to a 9-char string of 'R','Y','B','G','W','O' in Arduino index order (0..8).
        """
        name_to_idx = {v: k for k, v in FACE_NAMES.items()}
        for name, s in faces_dict.items():
            idx = name_to_idx[name.upper()]
            if len(s) != 9:
                raise ValueError(f"Face {name} must have 9 stickers, got {len(s)}")
            for i, ch in enumerate(s.upper()):
                if ch not in CHAR_TO_COLOR:
                    raise ValueError(f"Invalid color '{ch}' on face {name}[{i}]")
                self.rubik[idx][i] = CHAR_TO_COLOR[ch]

    def to_char_matrix(self) -> List[List[str]]:
        return [[COLOR_TO_CHAR[c] for c in face] for face in self.rubik]

    def to_char_dict(self) -> Dict[str, str]:
        return {
            FACE_NAMES[f]: "".join(COLOR_TO_CHAR[c] for c in self.rubik[f])
            for f in range(RF_TOTAL_FACES)
        }

    def is_solved(self) -> bool:
        for f in range(RF_TOTAL_FACES):
            c = self.rubik[f][4]
            if any(self.rubik[f][i] != c for i in range(9)):
                return False
        return True

    def validate_color_counts(self) -> Tuple[bool, str]:
        counts = {c: 0 for c in range(RC_TOTAL_COLORS)}
        for f in range(RF_TOTAL_FACES):
            for i in range(9):
                c = self.rubik[f][i]
                if c not in counts:
                    return False, f"Invalid color code {c} on {FACE_NAMES[f]}[{i}]"
                counts[c] += 1
        bad = {COLOR_TO_CHAR[k]: v for k, v in counts.items() if v != 9}
        if bad:
            return False, f"Each color must appear 9 times; mismatches: {bad}"
        centers = [self.rubik[f][4] for f in range(RF_TOTAL_FACES)]
        if len(set(centers)) != 6:
            return False, "All 6 face centers must be distinct colors."
        return True, "OK"

    def format_net(self) -> str:
        """Matches Cube::Print() in store_and_print.ino."""
        lines = []
        for i in (6, 3, 0):
            row = "".join(COLOR_TO_CHAR[self.rubik[RF_UP][i + k]] for k in range(3))
            lines.append("         " + row)
        lines.append("")
        for i in (6, 3, 0):
            parts = []
            for f in (RF_LEFT, RF_FRONT, RF_RIGHT, RF_BACK):
                parts.append("".join(COLOR_TO_CHAR[self.rubik[f][i + k]] for k in range(3)))
            lines.append("   ".join(parts))
        lines.append("")
        for i in (6, 3, 0):
            row = "".join(COLOR_TO_CHAR[self.rubik[RF_DOWN][i + k]] for k in range(3))
            lines.append("         " + row)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Rotations (exact 1:1 port of Rubik.ino lines 311-794)
    # -----------------------------------------------------------------------

    def _backup(self) -> List[List[int]]:
        return [row[:] for row in self.rubik]

    def rotate_front_ccw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_FRONT][0] = b[RF_FRONT][6]
        r[RF_FRONT][2] = b[RF_FRONT][0]
        r[RF_FRONT][8] = b[RF_FRONT][2]
        r[RF_FRONT][6] = b[RF_FRONT][8]
        r[RF_FRONT][1] = b[RF_FRONT][3]
        r[RF_FRONT][5] = b[RF_FRONT][1]
        r[RF_FRONT][7] = b[RF_FRONT][5]
        r[RF_FRONT][3] = b[RF_FRONT][7]
        r[RF_UP][0] = b[RF_RIGHT][6]
        r[RF_UP][1] = b[RF_RIGHT][3]
        r[RF_UP][2] = b[RF_RIGHT][0]
        r[RF_RIGHT][6] = b[RF_DOWN][8]
        r[RF_RIGHT][3] = b[RF_DOWN][7]
        r[RF_RIGHT][0] = b[RF_DOWN][6]
        r[RF_DOWN][8] = b[RF_LEFT][2]
        r[RF_DOWN][7] = b[RF_LEFT][5]
        r[RF_DOWN][6] = b[RF_LEFT][8]
        r[RF_LEFT][2] = b[RF_UP][0]
        r[RF_LEFT][5] = b[RF_UP][1]
        r[RF_LEFT][8] = b[RF_UP][2]

    def rotate_front_cw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_FRONT][0] = b[RF_FRONT][2]
        r[RF_FRONT][2] = b[RF_FRONT][8]
        r[RF_FRONT][8] = b[RF_FRONT][6]
        r[RF_FRONT][6] = b[RF_FRONT][0]
        r[RF_FRONT][1] = b[RF_FRONT][5]
        r[RF_FRONT][5] = b[RF_FRONT][7]
        r[RF_FRONT][7] = b[RF_FRONT][3]
        r[RF_FRONT][3] = b[RF_FRONT][1]
        r[RF_UP][0] = b[RF_LEFT][2]
        r[RF_UP][1] = b[RF_LEFT][5]
        r[RF_UP][2] = b[RF_LEFT][8]
        r[RF_RIGHT][6] = b[RF_UP][0]
        r[RF_RIGHT][3] = b[RF_UP][1]
        r[RF_RIGHT][0] = b[RF_UP][2]
        r[RF_DOWN][8] = b[RF_RIGHT][6]
        r[RF_DOWN][7] = b[RF_RIGHT][3]
        r[RF_DOWN][6] = b[RF_RIGHT][0]
        r[RF_LEFT][2] = b[RF_DOWN][8]
        r[RF_LEFT][5] = b[RF_DOWN][7]
        r[RF_LEFT][8] = b[RF_DOWN][6]

    def rotate_right_cw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_RIGHT][0] = b[RF_RIGHT][2]
        r[RF_RIGHT][2] = b[RF_RIGHT][8]
        r[RF_RIGHT][8] = b[RF_RIGHT][6]
        r[RF_RIGHT][6] = b[RF_RIGHT][0]
        r[RF_RIGHT][1] = b[RF_RIGHT][5]
        r[RF_RIGHT][5] = b[RF_RIGHT][7]
        r[RF_RIGHT][7] = b[RF_RIGHT][3]
        r[RF_RIGHT][3] = b[RF_RIGHT][1]
        r[RF_UP][2] = b[RF_FRONT][2]
        r[RF_UP][5] = b[RF_FRONT][5]
        r[RF_UP][8] = b[RF_FRONT][8]
        r[RF_BACK][6] = b[RF_UP][2]
        r[RF_BACK][3] = b[RF_UP][5]
        r[RF_BACK][0] = b[RF_UP][8]
        r[RF_DOWN][2] = b[RF_BACK][6]
        r[RF_DOWN][5] = b[RF_BACK][3]
        r[RF_DOWN][8] = b[RF_BACK][0]
        r[RF_FRONT][2] = b[RF_DOWN][2]
        r[RF_FRONT][5] = b[RF_DOWN][5]
        r[RF_FRONT][8] = b[RF_DOWN][8]

    def rotate_right_ccw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_RIGHT][0] = b[RF_RIGHT][6]
        r[RF_RIGHT][2] = b[RF_RIGHT][0]
        r[RF_RIGHT][8] = b[RF_RIGHT][2]
        r[RF_RIGHT][6] = b[RF_RIGHT][8]
        r[RF_RIGHT][1] = b[RF_RIGHT][3]
        r[RF_RIGHT][5] = b[RF_RIGHT][1]
        r[RF_RIGHT][7] = b[RF_RIGHT][5]
        r[RF_RIGHT][3] = b[RF_RIGHT][7]
        r[RF_UP][8] = b[RF_BACK][0]
        r[RF_UP][5] = b[RF_BACK][3]
        r[RF_UP][2] = b[RF_BACK][6]
        r[RF_FRONT][8] = b[RF_UP][8]
        r[RF_FRONT][5] = b[RF_UP][5]
        r[RF_FRONT][2] = b[RF_UP][2]
        r[RF_DOWN][8] = b[RF_FRONT][8]
        r[RF_DOWN][5] = b[RF_FRONT][5]
        r[RF_DOWN][2] = b[RF_FRONT][2]
        r[RF_BACK][0] = b[RF_DOWN][8]
        r[RF_BACK][3] = b[RF_DOWN][5]
        r[RF_BACK][6] = b[RF_DOWN][2]

    def rotate_up_cw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_UP][0] = b[RF_UP][2]
        r[RF_UP][2] = b[RF_UP][8]
        r[RF_UP][8] = b[RF_UP][6]
        r[RF_UP][6] = b[RF_UP][0]
        r[RF_UP][1] = b[RF_UP][5]
        r[RF_UP][5] = b[RF_UP][7]
        r[RF_UP][7] = b[RF_UP][3]
        r[RF_UP][3] = b[RF_UP][1]
        r[RF_BACK][8] = b[RF_LEFT][8]
        r[RF_BACK][7] = b[RF_LEFT][7]
        r[RF_BACK][6] = b[RF_LEFT][6]
        r[RF_RIGHT][8] = b[RF_BACK][8]
        r[RF_RIGHT][7] = b[RF_BACK][7]
        r[RF_RIGHT][6] = b[RF_BACK][6]
        r[RF_FRONT][8] = b[RF_RIGHT][8]
        r[RF_FRONT][7] = b[RF_RIGHT][7]
        r[RF_FRONT][6] = b[RF_RIGHT][6]
        r[RF_LEFT][8] = b[RF_FRONT][8]
        r[RF_LEFT][7] = b[RF_FRONT][7]
        r[RF_LEFT][6] = b[RF_FRONT][6]

    def rotate_up_ccw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_UP][0] = b[RF_UP][6]
        r[RF_UP][2] = b[RF_UP][0]
        r[RF_UP][8] = b[RF_UP][2]
        r[RF_UP][6] = b[RF_UP][8]
        r[RF_UP][1] = b[RF_UP][3]
        r[RF_UP][5] = b[RF_UP][1]
        r[RF_UP][7] = b[RF_UP][5]
        r[RF_UP][3] = b[RF_UP][7]
        r[RF_BACK][6] = b[RF_RIGHT][6]
        r[RF_BACK][7] = b[RF_RIGHT][7]
        r[RF_BACK][8] = b[RF_RIGHT][8]
        r[RF_LEFT][6] = b[RF_BACK][6]
        r[RF_LEFT][7] = b[RF_BACK][7]
        r[RF_LEFT][8] = b[RF_BACK][8]
        r[RF_FRONT][6] = b[RF_LEFT][6]
        r[RF_FRONT][7] = b[RF_LEFT][7]
        r[RF_FRONT][8] = b[RF_LEFT][8]
        r[RF_RIGHT][6] = b[RF_FRONT][6]
        r[RF_RIGHT][7] = b[RF_FRONT][7]
        r[RF_RIGHT][8] = b[RF_FRONT][8]

    def rotate_left_cw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_LEFT][0] = b[RF_LEFT][2]
        r[RF_LEFT][2] = b[RF_LEFT][8]
        r[RF_LEFT][8] = b[RF_LEFT][6]
        r[RF_LEFT][6] = b[RF_LEFT][0]
        r[RF_LEFT][3] = b[RF_LEFT][1]
        r[RF_LEFT][1] = b[RF_LEFT][5]
        r[RF_LEFT][5] = b[RF_LEFT][7]
        r[RF_LEFT][7] = b[RF_LEFT][3]
        r[RF_UP][6] = b[RF_BACK][2]
        r[RF_UP][3] = b[RF_BACK][5]
        r[RF_UP][0] = b[RF_BACK][8]
        r[RF_FRONT][6] = b[RF_UP][6]
        r[RF_FRONT][3] = b[RF_UP][3]
        r[RF_FRONT][0] = b[RF_UP][0]
        r[RF_DOWN][6] = b[RF_FRONT][6]
        r[RF_DOWN][3] = b[RF_FRONT][3]
        r[RF_DOWN][0] = b[RF_FRONT][0]
        r[RF_BACK][2] = b[RF_DOWN][6]
        r[RF_BACK][5] = b[RF_DOWN][3]
        r[RF_BACK][8] = b[RF_DOWN][0]

    def rotate_left_ccw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_LEFT][0] = b[RF_LEFT][6]
        r[RF_LEFT][2] = b[RF_LEFT][0]
        r[RF_LEFT][8] = b[RF_LEFT][2]
        r[RF_LEFT][6] = b[RF_LEFT][8]
        r[RF_LEFT][1] = b[RF_LEFT][3]
        r[RF_LEFT][5] = b[RF_LEFT][1]
        r[RF_LEFT][7] = b[RF_LEFT][5]
        r[RF_LEFT][3] = b[RF_LEFT][7]
        r[RF_UP][0] = b[RF_FRONT][0]
        r[RF_UP][3] = b[RF_FRONT][3]
        r[RF_UP][6] = b[RF_FRONT][6]
        r[RF_BACK][8] = b[RF_UP][0]
        r[RF_BACK][5] = b[RF_UP][3]
        r[RF_BACK][2] = b[RF_UP][6]
        r[RF_DOWN][0] = b[RF_BACK][8]
        r[RF_DOWN][3] = b[RF_BACK][5]
        r[RF_DOWN][6] = b[RF_BACK][2]
        r[RF_FRONT][0] = b[RF_DOWN][0]
        r[RF_FRONT][3] = b[RF_DOWN][3]
        r[RF_FRONT][6] = b[RF_DOWN][6]

    def rotate_down_cw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_DOWN][6] = b[RF_DOWN][0]
        r[RF_DOWN][8] = b[RF_DOWN][6]
        r[RF_DOWN][2] = b[RF_DOWN][8]
        r[RF_DOWN][0] = b[RF_DOWN][2]
        r[RF_DOWN][7] = b[RF_DOWN][3]
        r[RF_DOWN][5] = b[RF_DOWN][7]
        r[RF_DOWN][1] = b[RF_DOWN][5]
        r[RF_DOWN][3] = b[RF_DOWN][1]
        r[RF_FRONT][0] = b[RF_LEFT][0]
        r[RF_FRONT][1] = b[RF_LEFT][1]
        r[RF_FRONT][2] = b[RF_LEFT][2]
        r[RF_RIGHT][0] = b[RF_FRONT][0]
        r[RF_RIGHT][1] = b[RF_FRONT][1]
        r[RF_RIGHT][2] = b[RF_FRONT][2]
        r[RF_BACK][0] = b[RF_RIGHT][0]
        r[RF_BACK][1] = b[RF_RIGHT][1]
        r[RF_BACK][2] = b[RF_RIGHT][2]
        r[RF_LEFT][0] = b[RF_BACK][0]
        r[RF_LEFT][1] = b[RF_BACK][1]
        r[RF_LEFT][2] = b[RF_BACK][2]

    def rotate_down_ccw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_DOWN][6] = b[RF_DOWN][8]
        r[RF_DOWN][0] = b[RF_DOWN][6]
        r[RF_DOWN][2] = b[RF_DOWN][0]
        r[RF_DOWN][8] = b[RF_DOWN][2]
        r[RF_DOWN][7] = b[RF_DOWN][5]
        r[RF_DOWN][3] = b[RF_DOWN][7]
        r[RF_DOWN][1] = b[RF_DOWN][3]
        r[RF_DOWN][5] = b[RF_DOWN][1]
        r[RF_FRONT][2] = b[RF_RIGHT][2]
        r[RF_FRONT][1] = b[RF_RIGHT][1]
        r[RF_FRONT][0] = b[RF_RIGHT][0]
        r[RF_LEFT][2] = b[RF_FRONT][2]
        r[RF_LEFT][1] = b[RF_FRONT][1]
        r[RF_LEFT][0] = b[RF_FRONT][0]
        r[RF_BACK][2] = b[RF_LEFT][2]
        r[RF_BACK][1] = b[RF_LEFT][1]
        r[RF_BACK][0] = b[RF_LEFT][0]
        r[RF_RIGHT][2] = b[RF_BACK][2]
        r[RF_RIGHT][1] = b[RF_BACK][1]
        r[RF_RIGHT][0] = b[RF_BACK][0]

    def rotate_back_cw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_BACK][0] = b[RF_BACK][2]
        r[RF_BACK][6] = b[RF_BACK][0]
        r[RF_BACK][8] = b[RF_BACK][6]
        r[RF_BACK][2] = b[RF_BACK][8]
        r[RF_BACK][1] = b[RF_BACK][5]
        r[RF_BACK][3] = b[RF_BACK][1]
        r[RF_BACK][7] = b[RF_BACK][3]
        r[RF_BACK][5] = b[RF_BACK][7]
        r[RF_UP][8] = b[RF_RIGHT][2]
        r[RF_UP][7] = b[RF_RIGHT][5]
        r[RF_UP][6] = b[RF_RIGHT][8]
        r[RF_LEFT][6] = b[RF_UP][8]
        r[RF_LEFT][3] = b[RF_UP][7]
        r[RF_LEFT][0] = b[RF_UP][6]
        r[RF_DOWN][0] = b[RF_LEFT][6]
        r[RF_DOWN][1] = b[RF_LEFT][3]
        r[RF_DOWN][2] = b[RF_LEFT][0]
        r[RF_RIGHT][2] = b[RF_DOWN][0]
        r[RF_RIGHT][5] = b[RF_DOWN][1]
        r[RF_RIGHT][8] = b[RF_DOWN][2]

    def rotate_back_ccw(self) -> None:
        b = self._backup()
        r = self.rubik
        r[RF_BACK][0] = b[RF_BACK][6]
        r[RF_BACK][2] = b[RF_BACK][0]
        r[RF_BACK][8] = b[RF_BACK][2]
        r[RF_BACK][6] = b[RF_BACK][8]
        r[RF_BACK][1] = b[RF_BACK][3]
        r[RF_BACK][5] = b[RF_BACK][1]
        r[RF_BACK][7] = b[RF_BACK][5]
        r[RF_BACK][3] = b[RF_BACK][7]
        r[RF_UP][6] = b[RF_LEFT][0]
        r[RF_UP][7] = b[RF_LEFT][3]
        r[RF_UP][8] = b[RF_LEFT][6]
        r[RF_RIGHT][8] = b[RF_UP][6]
        r[RF_RIGHT][5] = b[RF_UP][7]
        r[RF_RIGHT][2] = b[RF_UP][8]
        r[RF_DOWN][2] = b[RF_RIGHT][8]
        r[RF_DOWN][1] = b[RF_RIGHT][5]
        r[RF_DOWN][0] = b[RF_RIGHT][2]
        r[RF_LEFT][0] = b[RF_DOWN][2]
        r[RF_LEFT][3] = b[RF_DOWN][1]
        r[RF_LEFT][6] = b[RF_DOWN][0]

    def rotate_Y(self) -> None:
        self.rotate_up_ccw()
        self.rotate_down_cw()
        b = self._backup()
        r = self.rubik
        for idx in (5, 4, 3):
            r[RF_FRONT][idx] = b[RF_LEFT][idx]
            r[RF_LEFT][idx] = b[RF_BACK][idx]
            r[RF_BACK][idx] = b[RF_RIGHT][idx]
            r[RF_RIGHT][idx] = b[RF_FRONT][idx]

    def rotate_y(self) -> None:
        self.rotate_up_cw()
        self.rotate_down_ccw()
        b = self._backup()
        r = self.rubik
        for idx in (5, 4, 3):
            r[RF_LEFT][idx] = b[RF_FRONT][idx]
            r[RF_BACK][idx] = b[RF_LEFT][idx]
            r[RF_RIGHT][idx] = b[RF_BACK][idx]
            r[RF_FRONT][idx] = b[RF_RIGHT][idx]

    def rotate_X(self) -> None:
        self.rotate_right_cw()
        self.rotate_left_ccw()
        b = self._backup()
        r = self.rubik
        r[RF_UP][7] = b[RF_FRONT][7]
        r[RF_UP][4] = b[RF_FRONT][4]
        r[RF_UP][1] = b[RF_FRONT][1]
        r[RF_BACK][1] = b[RF_UP][7]
        r[RF_BACK][4] = b[RF_UP][4]
        r[RF_BACK][7] = b[RF_UP][1]
        r[RF_DOWN][7] = b[RF_BACK][1]
        r[RF_DOWN][4] = b[RF_BACK][4]
        r[RF_DOWN][1] = b[RF_BACK][7]
        r[RF_FRONT][7] = b[RF_DOWN][7]
        r[RF_FRONT][4] = b[RF_DOWN][4]
        r[RF_FRONT][1] = b[RF_DOWN][1]

    def rotate_x(self) -> None:
        self.rotate_right_ccw()
        self.rotate_left_cw()
        b = self._backup()
        r = self.rubik
        r[RF_FRONT][7] = b[RF_UP][7]
        r[RF_FRONT][4] = b[RF_UP][4]
        r[RF_FRONT][1] = b[RF_UP][1]
        r[RF_UP][7] = b[RF_BACK][1]
        r[RF_UP][4] = b[RF_BACK][4]
        r[RF_UP][1] = b[RF_BACK][7]
        r[RF_BACK][1] = b[RF_DOWN][7]
        r[RF_BACK][4] = b[RF_DOWN][4]
        r[RF_BACK][7] = b[RF_DOWN][1]
        r[RF_DOWN][7] = b[RF_FRONT][7]
        r[RF_DOWN][4] = b[RF_FRONT][4]
        r[RF_DOWN][1] = b[RF_FRONT][1]

    def rotate_Z(self) -> None:
        self.rotate_front_ccw()
        self.rotate_back_cw()
        b = self._backup()
        r = self.rubik
        r[RF_UP][3] = b[RF_RIGHT][7]
        r[RF_UP][4] = b[RF_RIGHT][4]
        r[RF_UP][5] = b[RF_RIGHT][1]
        r[RF_RIGHT][7] = b[RF_DOWN][5]
        r[RF_RIGHT][4] = b[RF_DOWN][4]
        r[RF_RIGHT][1] = b[RF_DOWN][3]
        r[RF_DOWN][3] = b[RF_LEFT][7]
        r[RF_DOWN][4] = b[RF_LEFT][4]
        r[RF_DOWN][5] = b[RF_LEFT][1]
        r[RF_LEFT][7] = b[RF_UP][5]
        r[RF_LEFT][4] = b[RF_UP][4]
        r[RF_LEFT][1] = b[RF_UP][3]

    def rotate_z(self) -> None:
        self.rotate_front_cw()
        self.rotate_back_ccw()
        b = self._backup()
        r = self.rubik
        r[RF_RIGHT][7] = b[RF_UP][3]
        r[RF_RIGHT][4] = b[RF_UP][4]
        r[RF_RIGHT][1] = b[RF_UP][5]
        r[RF_DOWN][5] = b[RF_RIGHT][7]
        r[RF_DOWN][4] = b[RF_RIGHT][4]
        r[RF_DOWN][3] = b[RF_RIGHT][1]
        r[RF_LEFT][7] = b[RF_DOWN][3]
        r[RF_LEFT][4] = b[RF_DOWN][4]
        r[RF_LEFT][1] = b[RF_DOWN][5]
        r[RF_UP][5] = b[RF_LEFT][7]
        r[RF_UP][4] = b[RF_LEFT][4]
        r[RF_UP][3] = b[RF_LEFT][1]

    def rubik_rotate(self, rot: str) -> None:
        dispatch = {
            "f": self.rotate_front_ccw,
            "F": self.rotate_front_cw,
            "r": self.rotate_right_ccw,
            "R": self.rotate_right_cw,
            "u": self.rotate_up_ccw,
            "U": self.rotate_up_cw,
            "l": self.rotate_left_ccw,
            "L": self.rotate_left_cw,
            "d": self.rotate_down_ccw,
            "D": self.rotate_down_cw,
            "b": self.rotate_back_ccw,
            "B": self.rotate_back_cw,
            "Y": self.rotate_Y,
            "y": self.rotate_y,
            "X": self.rotate_X,
            "x": self.rotate_x,
            "Z": self.rotate_Z,
            "z": self.rotate_z,
        }
        if rot in dispatch:
            dispatch[rot]()

    def rotate_str(self, sequence: str) -> None:
        for ch in sequence:
            self.rubik_rotate(ch)
            self.resolution_str += ch

    def scramble(self, rotations_count: int = 20, seed: Optional[int] = None) -> str:
        rng = random.Random(seed)
        rotations = ["f", "F", "r", "R", "u", "U", "l", "L", "d", "D", "b", "B"]
        seq = []
        for _ in range(rotations_count):
            mv = rng.choice(rotations)
            self.rubik_rotate(mv)
            seq.append(mv)
        return "".join(seq)

    # -----------------------------------------------------------------------
    # Self-Orientation (rubik_orient_SolveCross.ino)
    # -----------------------------------------------------------------------

    def self_orient(self) -> str:
        moves = []
        r = self.rubik
        if r[RF_FRONT][4] == RC_WHITE:
            self.rotate_x()
            moves.append("x")
        elif r[RF_BACK][4] == RC_WHITE:
            self.rotate_X()
            moves.append("X")
        elif r[RF_RIGHT][4] == RC_WHITE:
            self.rotate_z()
            moves.append("z")
        elif r[RF_LEFT][4] == RC_WHITE:
            self.rotate_Z()
            moves.append("Z")
        elif r[RF_UP][4] == RC_WHITE:
            self.rotate_X()
            self.rotate_X()
            moves.extend(["X", "X"])

        if r[RF_LEFT][4] == RC_RED:
            self.rotate_Y()
            moves.append("Y")
        elif r[RF_RIGHT][4] == RC_RED:
            self.rotate_y()
            moves.append("y")
        elif r[RF_BACK][4] == RC_RED:
            self.rotate_Y()
            self.rotate_Y()
            moves.extend(["Y", "Y"])
        return "".join(moves)

    # -----------------------------------------------------------------------
    # White Cross (rubik_orient_SolveCross.ino)
    # -----------------------------------------------------------------------

    def solve_white_cross_1(self) -> None:
        r = self.rubik
        if r[RF_DOWN][1] == RC_WHITE:
            m = {RC_RED: "dd", RC_BLUE: "D", RC_ORANGE: "", RC_GREEN: "d"}.get(r[RF_BACK][1], "")
            self.rotate_str(m)
        if r[RF_DOWN][3] == RC_WHITE:
            m = {RC_RED: "D", RC_BLUE: "", RC_ORANGE: "d", RC_GREEN: "dd"}.get(r[RF_LEFT][1], "")
            self.rotate_str(m)
        if r[RF_DOWN][5] == RC_WHITE:
            m = {RC_RED: "d", RC_BLUE: "dd", RC_ORANGE: "D", RC_GREEN: ""}.get(r[RF_RIGHT][1], "")
            self.rotate_str(m)
        if r[RF_DOWN][7] == RC_WHITE:
            m = {RC_RED: "", RC_BLUE: "d", RC_ORANGE: "dd", RC_GREEN: "D"}.get(r[RF_FRONT][1], "")
            self.rotate_str(m)

    def solve_white_cross_2(self, depth: int = 0) -> None:
        if depth > 20:
            return
        r = self.rubik
        rot = 0
        if r[RF_UP][1] == RC_WHITE:
            side_color = r[RF_FRONT][7]
            rot -= 0
        elif r[RF_UP][3] == RC_WHITE:
            side_color = r[RF_LEFT][7]
            rot -= 3
        elif r[RF_UP][5] == RC_WHITE:
            side_color = r[RF_RIGHT][7]
            rot -= 1
        elif r[RF_UP][7] == RC_WHITE:
            side_color = r[RF_BACK][7]
            rot -= 2
        else:
            return

        rot += {RC_RED: 0, RC_GREEN: 1, RC_ORANGE: 2, RC_BLUE: 3}.get(side_color, 0)
        if rot == 0:
            self.rotate_str("")
        elif rot in (-3, 1):
            self.rotate_str("u")
        elif rot in (2, -2):
            self.rotate_str("UU")
        elif rot in (-1, 3):
            self.rotate_str("U")

        face_rot = {RC_RED: "FF", RC_GREEN: "RR", RC_ORANGE: "BB", RC_BLUE: "LL"}.get(side_color, "")
        self.rotate_str(face_rot)
        self.solve_white_cross_2(depth + 1)

    def solve_white_cross_3(self, depth: int = 0) -> None:
        if depth > 20:
            return
        r = self.rubik
        if r[RF_FRONT][7] == RC_WHITE:
            m = {RC_RED: "urFR", RC_BLUE: "fLF", RC_ORANGE: "uRbr", RC_GREEN: "Frf"}.get(r[RF_UP][1], "")
            self.rotate_str(m)
        elif r[RF_LEFT][7] == RC_WHITE:
            m = {RC_RED: "Lfl", RC_BLUE: "ufLF", RC_ORANGE: "lBL", RC_GREEN: "uFrf"}.get(r[RF_UP][3], "")
            self.rotate_str(m)
        elif r[RF_RIGHT][7] == RC_WHITE:
            m = {RC_RED: "rFR", RC_BLUE: "UfLF", RC_ORANGE: "Rbr", RC_GREEN: "UFrf"}.get(r[RF_UP][5], "")
            self.rotate_str(m)
        elif r[RF_BACK][7] == RC_WHITE:
            m = {RC_RED: "uLfl", RC_BLUE: "Blb", RC_ORANGE: "ulBL", RC_GREEN: "bRB"}.get(r[RF_UP][7], "")
            self.rotate_str(m)
        else:
            return
        self.solve_white_cross_2()
        self.solve_white_cross_3(depth + 1)

    def solve_white_cross_4(self, depth: int = 0) -> None:
        if depth > 20:
            return
        r = self.rubik
        if r[RF_FRONT][3] == RC_WHITE and r[RF_LEFT][5] != RC_BLUE:
            self.rotate_str("lUL")
        elif r[RF_FRONT][3] == RC_WHITE and r[RF_LEFT][5] == RC_BLUE:
            self.rotate_str("L")
        elif r[RF_FRONT][5] == RC_WHITE and r[RF_RIGHT][3] != RC_GREEN:
            self.rotate_str("RUr")
        elif r[RF_FRONT][5] == RC_WHITE and r[RF_RIGHT][3] == RC_GREEN:
            self.rotate_str("r")
        elif r[RF_BACK][3] == RC_WHITE and r[RF_RIGHT][5] != RC_GREEN:
            self.rotate_str("ruR")
        elif r[RF_BACK][3] == RC_WHITE and r[RF_RIGHT][5] == RC_GREEN:
            self.rotate_str("R")
        elif r[RF_BACK][5] == RC_WHITE and r[RF_LEFT][3] != RC_BLUE:
            self.rotate_str("LUl")
        elif r[RF_BACK][5] == RC_WHITE and r[RF_LEFT][3] == RC_BLUE:
            self.rotate_str("l")
        else:
            return
        self.solve_white_cross_2()
        self.solve_white_cross_3()
        self.solve_white_cross_4(depth + 1)

    def solve_white_cross_5(self, depth: int = 0) -> None:
        if depth > 20:
            return
        r = self.rubik
        if r[RF_LEFT][3] == RC_WHITE and r[RF_BACK][5] != RC_ORANGE:
            self.rotate_str("buB")
        elif r[RF_LEFT][3] == RC_WHITE and r[RF_BACK][5] == RC_ORANGE:
            self.rotate_str("B")
        elif r[RF_LEFT][5] == RC_WHITE and r[RF_FRONT][3] != RC_RED:
            self.rotate_str("FUf")
        elif r[RF_LEFT][5] == RC_WHITE and r[RF_FRONT][3] == RC_RED:
            self.rotate_str("f")
        elif r[RF_RIGHT][3] == RC_WHITE and r[RF_FRONT][5] != RC_RED:
            self.rotate_str("fuF")
        elif r[RF_RIGHT][3] == RC_WHITE and r[RF_FRONT][5] == RC_RED:
            self.rotate_str("F")
        elif r[RF_RIGHT][5] == RC_WHITE and r[RF_BACK][3] != RC_ORANGE:
            self.rotate_str("BUb")
        elif r[RF_RIGHT][5] == RC_WHITE and r[RF_BACK][3] == RC_ORANGE:
            self.rotate_str("b")
        else:
            return
        self.solve_white_cross_2()
        self.solve_white_cross_3()
        self.solve_white_cross_4()
        self.solve_white_cross_5(depth + 1)

    def solve_white_cross_6(self) -> None:
        r = self.rubik
        if r[RF_FRONT][1] == RC_WHITE:
            self.rotate_str("FF")
        if r[RF_RIGHT][1] == RC_WHITE:
            self.rotate_str("RR")
        if r[RF_LEFT][1] == RC_WHITE:
            self.rotate_str("LL")
        if r[RF_BACK][1] == RC_WHITE:
            self.rotate_str("bb")
        self.solve_white_cross_3()
        self.solve_white_cross_4()
        self.solve_white_cross_5()

    # -----------------------------------------------------------------------
    # F2L (rubik_SolveF2L.ino)
    # -----------------------------------------------------------------------

    def _f2l_corner_exception(self, front_color: int, right_color: int) -> None:
        r = self.rubik
        fc, rc = front_color, right_color
        if (
            (r[RF_UP][8] == RC_WHITE and r[RF_RIGHT][8] == rc and r[RF_BACK][6] == fc)
            or (r[RF_UP][8] == rc and r[RF_RIGHT][8] == fc and r[RF_BACK][6] == RC_WHITE)
            or (r[RF_UP][8] == fc and r[RF_RIGHT][8] == RC_WHITE and r[RF_BACK][6] == rc)
        ):
            self.rotate_str("U")
        elif (
            (r[RF_UP][6] == RC_WHITE and r[RF_BACK][8] == rc and r[RF_LEFT][6] == fc)
            or (r[RF_UP][6] == rc and r[RF_BACK][8] == fc and r[RF_LEFT][6] == RC_WHITE)
            or (r[RF_UP][6] == fc and r[RF_BACK][8] == RC_WHITE and r[RF_LEFT][6] == rc)
        ):
            self.rotate_str("UU")
        elif (
            (r[RF_UP][0] == RC_WHITE and r[RF_FRONT][6] == fc and r[RF_LEFT][8] == rc)
            or (r[RF_UP][0] == fc and r[RF_FRONT][6] == rc and r[RF_LEFT][8] == RC_WHITE)
            or (r[RF_UP][0] == rc and r[RF_FRONT][6] == RC_WHITE and r[RF_LEFT][8] == fc)
        ):
            self.rotate_str("u")
        elif (
            (r[RF_DOWN][2] == RC_WHITE and r[RF_RIGHT][2] == fc and r[RF_BACK][0] == rc)
            or (r[RF_DOWN][2] == fc and r[RF_RIGHT][2] == rc and r[RF_BACK][0] == RC_WHITE)
            or (r[RF_DOWN][2] == rc and r[RF_RIGHT][2] == RC_WHITE and r[RF_BACK][0] == fc)
        ):
            self.rotate_str("rUURu")
        elif (
            (r[RF_DOWN][0] == RC_WHITE and r[RF_BACK][2] == fc and r[RF_LEFT][0] == rc)
            or (r[RF_DOWN][0] == fc and r[RF_BACK][2] == rc and r[RF_LEFT][0] == RC_WHITE)
            or (r[RF_DOWN][0] == rc and r[RF_BACK][2] == RC_WHITE and r[RF_LEFT][0] == fc)
        ):
            self.rotate_str("LUUl")
        elif (
            (r[RF_DOWN][6] == RC_WHITE and r[RF_LEFT][2] == fc and r[RF_FRONT][0] == rc)
            or (r[RF_DOWN][6] == fc and r[RF_LEFT][2] == rc and r[RF_FRONT][0] == RC_WHITE)
            or (r[RF_DOWN][6] == rc and r[RF_LEFT][2] == RC_WHITE and r[RF_FRONT][0] == fc)
        ):
            self.rotate_str("luL")

    def _f2l_edge_exception(self, front_color: int, right_color: int) -> None:
        r = self.rubik
        fc, rc = front_color, right_color
        if (r[RF_RIGHT][5] == fc and r[RF_BACK][3] == rc) or (r[RF_RIGHT][5] == rc and r[RF_BACK][3] == fc):
            self.rotate_str("ruR")
        elif (r[RF_BACK][5] == fc and r[RF_LEFT][3] == rc) or (r[RF_BACK][5] == rc and r[RF_LEFT][3] == fc):
            self.rotate_str("LulU")
        elif (r[RF_LEFT][5] == fc and r[RF_FRONT][3] == rc) or (r[RF_LEFT][5] == rc and r[RF_FRONT][3] == fc):
            self.rotate_str("luLU")

    def solve_f2l_slot(self, front_color: int, right_color: int) -> None:
        r = self.rubik
        fc, rc = front_color, right_color
        self._f2l_corner_exception(fc, rc)

        c = -1
        if r[RF_DOWN][8] == RC_WHITE and r[RF_FRONT][2] == fc and r[RF_RIGHT][0] == rc:
            c = 0
        elif r[RF_DOWN][8] == rc and r[RF_FRONT][2] == RC_WHITE and r[RF_RIGHT][0] == fc:
            c = 1
        elif r[RF_DOWN][8] == fc and r[RF_FRONT][2] == rc and r[RF_RIGHT][0] == RC_WHITE:
            c = 2
        elif r[RF_UP][2] == fc and r[RF_FRONT][8] == RC_WHITE and r[RF_RIGHT][6] == rc:
            c = 3
        elif r[RF_UP][2] == RC_WHITE and r[RF_FRONT][8] == rc and r[RF_RIGHT][6] == fc:
            c = 4
        elif r[RF_UP][2] == rc and r[RF_FRONT][8] == fc and r[RF_RIGHT][6] == RC_WHITE:
            c = 5

        self._f2l_edge_exception(fc, rc)

        e = -1
        if r[RF_FRONT][5] == fc and r[RF_RIGHT][3] == rc:
            e = 0
        elif r[RF_FRONT][5] == rc and r[RF_RIGHT][3] == fc:
            e = 1
        elif r[RF_UP][1] == fc and r[RF_FRONT][7] == rc:
            e = 2
        elif r[RF_UP][1] == rc and r[RF_FRONT][7] == fc:
            e = 3
        elif r[RF_UP][5] == fc and r[RF_RIGHT][7] == rc:
            e = 4
        elif r[RF_UP][5] == rc and r[RF_RIGHT][7] == fc:
            e = 5
        elif r[RF_UP][7] == fc and r[RF_BACK][7] == rc:
            e = 6
        elif r[RF_UP][7] == rc and r[RF_BACK][7] == fc:
            e = 7
        elif r[RF_UP][3] == fc and r[RF_LEFT][7] == rc:
            e = 8
        elif r[RF_UP][3] == rc and r[RF_LEFT][7] == fc:
            e = 9

        if c < 0 or e < 0:
            raise ValueError(
                f"Invalid cube state during F2L slot ({COLOR_TO_CHAR[fc]}/{COLOR_TO_CHAR[rc]}): c={c}, e={e}"
            )
        self.rotate_str(F2L_LUT[c * 10 + e])

    def solve_f2l(self) -> None:
        self.solve_f2l_slot(RC_RED, RC_GREEN)
        self.rotate_str("Y")
        self.solve_f2l_slot(RC_BLUE, RC_RED)
        self.rotate_str("Y")
        self.solve_f2l_slot(RC_ORANGE, RC_BLUE)
        self.rotate_str("Y")
        self.solve_f2l_slot(RC_GREEN, RC_ORANGE)
        self.rotate_str("Y")

    # -----------------------------------------------------------------------
    # OLL (rubik_SolveOLL.ino)
    # -----------------------------------------------------------------------

    def solve_oll(self) -> int:
        r = self.rubik
        c = -1
        e = -1
        for _ in range(4):
            if (
                r[RF_RIGHT][6] == RC_YELLOW and r[RF_RIGHT][8] == RC_YELLOW
                and r[RF_LEFT][6] == RC_YELLOW and r[RF_LEFT][8] == RC_YELLOW
            ):
                c = 0
                break
            elif (
                r[RF_LEFT][6] == RC_YELLOW and r[RF_LEFT][8] == RC_YELLOW
                and r[RF_FRONT][8] == RC_YELLOW and r[RF_BACK][6] == RC_YELLOW
            ):
                c = 1
                break
            elif (
                r[RF_UP][0] == RC_YELLOW and r[RF_FRONT][8] == RC_YELLOW
                and r[RF_RIGHT][8] == RC_YELLOW and r[RF_BACK][8] == RC_YELLOW
            ):
                c = 2
                break
            elif (
                r[RF_UP][0] == RC_YELLOW and r[RF_RIGHT][6] == RC_YELLOW
                and r[RF_LEFT][6] == RC_YELLOW and r[RF_BACK][6] == RC_YELLOW
            ):
                c = 3
                break
            elif (
                r[RF_UP][2] == RC_YELLOW and r[RF_UP][8] == RC_YELLOW
                and r[RF_LEFT][6] == RC_YELLOW and r[RF_LEFT][8] == RC_YELLOW
            ):
                c = 4
                break
            elif (
                r[RF_UP][2] == RC_YELLOW and r[RF_UP][8] == RC_YELLOW
                and r[RF_BACK][8] == RC_YELLOW and r[RF_FRONT][6] == RC_YELLOW
            ):
                c = 5
                break
            elif (
                r[RF_UP][0] == RC_YELLOW and r[RF_UP][8] == RC_YELLOW
                and r[RF_FRONT][8] == RC_YELLOW and r[RF_LEFT][6] == RC_YELLOW
            ):
                c = 6
                break
            elif (
                r[RF_UP][0] == RC_YELLOW and r[RF_UP][2] == RC_YELLOW
                and r[RF_UP][6] == RC_YELLOW and r[RF_UP][8] == RC_YELLOW
            ):
                c = 7
                break
            else:
                self.rotate_str("U")

        if (
            r[RF_FRONT][7] == RC_YELLOW and r[RF_RIGHT][7] == RC_YELLOW
            and r[RF_BACK][7] == RC_YELLOW and r[RF_LEFT][7] == RC_YELLOW
        ):
            e = 0
        elif (
            r[RF_UP][3] == RC_YELLOW and r[RF_UP][5] == RC_YELLOW
            and r[RF_FRONT][7] == RC_YELLOW and r[RF_BACK][7] == RC_YELLOW
        ):
            e = 1
        elif (
            r[RF_UP][1] == RC_YELLOW and r[RF_UP][7] == RC_YELLOW
            and r[RF_RIGHT][7] == RC_YELLOW and r[RF_LEFT][7] == RC_YELLOW
        ):
            e = 2
        elif (
            r[RF_UP][3] == RC_YELLOW and r[RF_UP][7] == RC_YELLOW
            and r[RF_FRONT][7] == RC_YELLOW and r[RF_RIGHT][7] == RC_YELLOW
        ):
            e = 3
        elif (
            r[RF_UP][3] == RC_YELLOW and r[RF_UP][1] == RC_YELLOW
            and r[RF_RIGHT][7] == RC_YELLOW and r[RF_BACK][7] == RC_YELLOW
        ):
            e = 4
        elif (
            r[RF_UP][1] == RC_YELLOW and r[RF_UP][5] == RC_YELLOW
            and r[RF_LEFT][7] == RC_YELLOW and r[RF_BACK][7] == RC_YELLOW
        ):
            e = 5
        elif (
            r[RF_UP][5] == RC_YELLOW and r[RF_UP][7] == RC_YELLOW
            and r[RF_FRONT][7] == RC_YELLOW and r[RF_LEFT][7] == RC_YELLOW
        ):
            e = 6
        elif (
            r[RF_UP][1] == RC_YELLOW and r[RF_UP][5] == RC_YELLOW
            and r[RF_UP][7] == RC_YELLOW and r[RF_UP][3] == RC_YELLOW
        ):
            e = 7

        if c < 0 or e < 0:
            raise ValueError(f"Unrecognized OLL pattern (unsolvable cube state): e={e}, c={c}")

        idx = e * 8 + c
        self.rotate_str(OLL_LUT[idx])
        return idx

    # -----------------------------------------------------------------------
    # PLL + AUF (rubik_SolvePLL_and_AUF.ino)
    # -----------------------------------------------------------------------

    def auf(self) -> None:
        m = {
            RC_RED: "",
            RC_BLUE: "U",
            RC_ORANGE: "uu",
            RC_GREEN: "u",
        }.get(self.rubik[RF_FRONT][7], "")
        self.rotate_str(m)

    def solve_pll(self) -> int:
        r = self.rubik
        caso = -1
        if (
            r[RF_FRONT][6] == r[RF_FRONT][7] == r[RF_FRONT][8]
            and r[RF_LEFT][6] == r[RF_LEFT][7] == r[RF_LEFT][8]
            and r[RF_BACK][6] == r[RF_BACK][7] == r[RF_BACK][8]
            and r[RF_RIGHT][6] == r[RF_RIGHT][7] == r[RF_RIGHT][8]
        ):
            caso = 0
        else:
            for _ in range(4):
                if r[RF_LEFT][6] == r[RF_LEFT][8] and r[RF_FRONT][6] != r[RF_FRONT][8]:
                    if r[RF_FRONT][6] == r[RF_FRONT][7] and r[RF_BACK][7] == r[RF_BACK][8]:
                        self.rotate_str("RUrurFRRuruRUrf")
                        caso = 1
                        break
                    elif (
                        r[RF_LEFT][7] == r[RF_LEFT][6]
                        and r[RF_FRONT][8] == r[RF_BACK][6] == r[RF_RIGHT][7]
                    ):
                        self.rotate_str("rufRUrurFRRuruRUrUR")
                        caso = 2
                        break
                    elif r[RF_FRONT][7] == r[RF_FRONT][8] and r[RF_RIGHT][6] == r[RF_RIGHT][7]:
                        self.rotate_str("UrFrBBRfrBBrr")
                        caso = 3
                        break
                    elif r[RF_RIGHT][7] == r[RF_RIGHT][8] and r[RF_BACK][6] == r[RF_BACK][7]:
                        self.rotate_str("uRbRffrBRffRR")
                        caso = 4
                        break
                    elif (
                        r[RF_FRONT][6] == r[RF_FRONT][7]
                        and r[RF_FRONT][8] == r[RF_BACK][6] == r[RF_RIGHT][7]
                    ):
                        self.rotate_str("RUrfRUUrUUrFRURuur")
                        caso = 5
                        break
                    elif (
                        r[RF_BACK][7] == r[RF_BACK][8]
                        and r[RF_FRONT][8] == r[RF_BACK][6] == r[RF_RIGHT][7]
                    ):
                        self.rotate_str("urUUrdRurDRURuruR")
                        caso = 6
                        break
                    elif (
                        r[RF_FRONT][6] == r[RF_FRONT][7]
                        and r[RF_RIGHT][6] == r[RF_RIGHT][7]
                        and r[RF_BACK][6] == r[RF_BACK][7]
                        and r[RF_LEFT][6] == r[RF_LEFT][8]
                    ):
                        self.rotate_str("UUluLFluLULfllUL")
                        caso = 7
                        break
                    elif (
                        r[RF_FRONT][7] == r[RF_FRONT][8]
                        and r[RF_RIGHT][7] == r[RF_RIGHT][8]
                        and r[RF_BACK][7] == r[RF_BACK][8]
                        and r[RF_LEFT][6] == r[RF_LEFT][8]
                    ):
                        self.rotate_str("RUUruRUUlUruL")
                        caso = 8
                        break
                    elif r[RF_FRONT][7] == r[RF_FRONT][8]:
                        self.rotate_str("dRRUrUruRuRRuDrUR")
                        caso = 9
                        break
                    elif r[RF_BACK][6] == r[RF_BACK][7]:
                        self.rotate_str("DRRuRuRUrUrrUdRur")
                        caso = 10
                        break
                    elif r[RF_RIGHT][6] == r[RF_RIGHT][7]:
                        self.rotate_str("RUruDRRuRurUrUrrd")
                        caso = 11
                        break
                    elif r[RF_RIGHT][7] == r[RF_RIGHT][8]:
                        self.rotate_str("DruRUdRRUrURuRurr")
                        caso = 12
                        break

                if (
                    r[RF_FRONT][6] == r[RF_FRONT][7]
                    and r[RF_RIGHT][7] == r[RF_RIGHT][8]
                    and r[RF_FRONT][8] == r[RF_BACK][6]
                ):
                    self.rotate_str("FUfrrFufuRRUrrURR")
                    caso = 13
                    break
                elif (
                    r[RF_FRONT][6] == r[RF_FRONT][7]
                    and r[RF_LEFT][7] == r[RF_LEFT][8]
                    and r[RF_FRONT][8] == r[RF_BACK][6]
                    and r[RF_RIGHT][6] != r[RF_RIGHT][7]
                ):
                    self.rotate_str("rUURuuLurUlULuRUl")
                    caso = 14
                    break
                elif (
                    r[RF_FRONT][7] == RC_RED
                    and r[RF_LEFT][7] == RC_BLUE
                    and r[RF_BACK][7] == RC_ORANGE
                    and r[RF_RIGHT][7] == RC_GREEN
                    and r[RF_FRONT][8] == r[RF_BACK][6]
                ):
                    # Fixed E-perm orientation check from rubik_SolvePLL_and_AUF.ino:
                    # Distinguish the two 90-deg E-perm orientations by FRONT[8] corner color.
                    if r[RF_FRONT][8] == RC_GREEN:
                        self.rotate_str("UrUlddLuRlUrDDRuL")
                    else:
                        self.rotate_str("rUlddLuRlUrDDRuL")
                    caso = 15
                    break
                elif (
                    r[RF_LEFT][7] == r[RF_LEFT][8]
                    and r[RF_BACK][7] == r[RF_BACK][8]
                    and r[RF_RIGHT][7] == r[RF_RIGHT][8]
                    and r[RF_FRONT][7] == r[RF_FRONT][8]
                    and r[RF_RIGHT][6] != r[RF_RIGHT][8]
                    and r[RF_FRONT][6] != r[RF_FRONT][8]
                    and r[RF_BACK][6] != r[RF_BACK][8]
                    and r[RF_LEFT][6] != r[RF_LEFT][8]
                ):
                    self.rotate_str("LuRUUlUrLuRuulUr")
                    caso = 16
                    break
                elif (
                    r[RF_LEFT][6] == r[RF_LEFT][7]
                    and r[RF_BACK][6] == r[RF_BACK][7]
                    and r[RF_RIGHT][6] == r[RF_RIGHT][7]
                    and r[RF_FRONT][6] == r[RF_FRONT][7]
                    and r[RF_RIGHT][6] != r[RF_RIGHT][8]
                    and r[RF_FRONT][6] != r[RF_FRONT][8]
                    and r[RF_BACK][6] != r[RF_BACK][8]
                    and r[RF_LEFT][6] != r[RF_LEFT][8]
                ):
                    self.rotate_str("lUrUULuRlUrUULuR")
                    caso = 17
                    break

                if (
                    r[RF_LEFT][6] == r[RF_LEFT][8]
                    and r[RF_BACK][6] == r[RF_BACK][8]
                    and r[RF_RIGHT][6] == r[RF_RIGHT][8]
                    and r[RF_FRONT][6] == r[RF_FRONT][8]
                ):
                    if r[RF_FRONT][7] == r[RF_FRONT][6] and r[RF_LEFT][7] == r[RF_BACK][8]:
                        self.rotate_str("rUrururURUrr")
                        caso = 18
                        break
                    elif r[RF_FRONT][7] == r[RF_FRONT][6] and r[RF_LEFT][7] == r[RF_RIGHT][8]:
                        self.rotate_str("LuLULULuluLL")
                        caso = 19
                        break
                    elif r[RF_FRONT][7] == r[RF_BACK][6] and r[RF_LEFT][7] == r[RF_RIGHT][6]:
                        self.rotate_str("rrUURuuRRuuRRuuRUUrr")
                        caso = 20
                        break
                    elif r[RF_FRONT][7] == r[RF_RIGHT][6] and r[RF_LEFT][7] == r[RF_BACK][6]:
                        self.rotate_str("RUrUrurURuruRRUR")
                        caso = 21
                        break

                self.rotate_str("u")

        self.auf()
        return caso

    # -----------------------------------------------------------------------
    # Move Optimizer (ProcessStr in Rubik.ino lines 152-216)
    # -----------------------------------------------------------------------

    @staticmethod
    def process_str(seq: str) -> str:
        s = list(seq)
        inverse_pairs = {
            ("r", "R"), ("R", "r"),
            ("l", "L"), ("L", "l"),
            ("f", "F"), ("F", "f"),
            ("b", "B"), ("B", "b"),
            ("u", "U"), ("U", "u"),
            ("d", "D"), ("D", "d"),
            ("Y", "y"), ("y", "Y"),
        }
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(s):
                # 4 identical moves -> remove all 4
                if i + 3 < len(s) and s[i] == s[i + 1] == s[i + 2] == s[i + 3]:
                    del s[i : i + 4]
                    changed = True
                    continue
                # Inverse pair -> cancel both
                if i + 1 < len(s) and (s[i], s[i + 1]) in inverse_pairs:
                    del s[i : i + 2]
                    changed = True
                    continue
                # 3 identical moves -> single inverse move
                if i + 2 < len(s) and s[i] == s[i + 1] == s[i + 2]:
                    ch = s[i]
                    s[i] = ch.lower() if ch.isupper() else ch.upper()
                    del s[i + 1 : i + 3]
                    changed = True
                    continue
                i += 1
        return "".join(s)

    # -----------------------------------------------------------------------
    # Full Solve Pipeline (matching setup() + SolveSequence() in Arduino)
    # -----------------------------------------------------------------------

    def solve(self) -> SolveResult:
        initial_char = self.to_char_matrix()
        ok, msg = self.validate_color_counts()
        if not ok:
            return SolveResult(
                solved=False,
                orient_moves="",
                raw_solve_moves="",
                optimized_solve_moves="",
                full_moves="",
                stage_breakdown=[],
                initial_state=initial_char,
                final_state=initial_char,
                error=msg,
            )

        backup_initial = self.clone_state()
        self.resolution_str = ""
        stages: List[SolveStageRecord] = []

        try:
            # 1. Self-orient (executes immediately on robot before SolveSequence)
            orient_moves = self.self_orient()
            stages.append(SolveStageRecord("Orient", orient_moves))

            # 2. White Cross
            start = len(self.resolution_str)
            self.solve_white_cross_1()
            self.solve_white_cross_2()
            self.solve_white_cross_3()
            self.solve_white_cross_4()
            self.solve_white_cross_5()
            self.solve_white_cross_6()
            stages.append(SolveStageRecord("White Cross", self.resolution_str[start:]))

            # 3. F2L
            start = len(self.resolution_str)
            self.solve_f2l()
            stages.append(SolveStageRecord("F2L", self.resolution_str[start:]))

            # 4. OLL
            start = len(self.resolution_str)
            self.solve_oll()
            stages.append(SolveStageRecord("OLL", self.resolution_str[start:]))

            # 5. PLL + AUF
            start = len(self.resolution_str)
            self.solve_pll()
            stages.append(SolveStageRecord("PLL & AUF", self.resolution_str[start:]))

            raw_solve = self.resolution_str
            optimized_solve = self.process_str(raw_solve)
            full_moves = orient_moves + optimized_solve

            # Replay from initial state using full_moves to verify exact equivalence
            self.set_state(backup_initial)
            for mv in full_moves:
                self.rubik_rotate(mv)

            solved = self.is_solved()
            return SolveResult(
                solved=solved,
                orient_moves=orient_moves,
                raw_solve_moves=raw_solve,
                optimized_solve_moves=optimized_solve,
                full_moves=full_moves,
                stage_breakdown=stages,
                initial_state=initial_char,
                final_state=self.to_char_matrix(),
                error=None if solved else "Cube did not reach solved state (check sticker orientation/parity).",
            )
        except Exception as exc:
            self.set_state(backup_initial)
            return SolveResult(
                solved=False,
                orient_moves="",
                raw_solve_moves="",
                optimized_solve_moves="",
                full_moves="",
                stage_breakdown=stages,
                initial_state=initial_char,
                final_state=initial_char,
                error=str(exc),
            )
