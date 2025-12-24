"""
Rubick's cube logic and structure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from enum import Enum, auto
from typing import Dict, List, Optional
from dataclasses import dataclass

import matplotlib
matplotlib.use("Qt5Agg")


class Face(Enum):
    FRONT = 'F'
    BACK = 'B'
    UP = 'U'
    DOWN = 'D'
    LEFT = 'L'
    RIGHT = 'R'


class MoveSpecifier(Enum):
    CLOCKWISE = auto()
    COUNTERCLOCKWISE = auto()
    HALF_TURN = auto()


@dataclass(frozen=True)
class Move:
    face: Face
    specifier: MoveSpecifier

    def __str__(self) -> str:
        spec = ""
        if self.specifier == MoveSpecifier.COUNTERCLOCKWISE:
            spec = "'"
        elif self.specifier == MoveSpecifier.HALF_TURN:
            spec = "2"
        return f"{self.face.value}{spec}"

    def __repr__(self) -> str:
        return str(self)

    def get_inverse(self) -> 'Move':
        """Get the inverse of this move."""
        if self.specifier == MoveSpecifier.CLOCKWISE:
            inverse_specifier = MoveSpecifier.COUNTERCLOCKWISE
        elif self.specifier == MoveSpecifier.COUNTERCLOCKWISE:
            inverse_specifier = MoveSpecifier.CLOCKWISE
        else:
            inverse_specifier = MoveSpecifier.HALF_TURN

        return Move(self.face, inverse_specifier)


faces = list(Face)
specs = list(MoveSpecifier)
moves = [Move(face, spec) for face in faces for spec in specs]
moves_idx = {move: i for i, move in enumerate(moves)}


FACE_COLORS: Dict[Face, str] = {
    Face.FRONT: 'green',
    Face.BACK: 'blue',
    Face.UP: 'white',
    Face.DOWN: 'yellow',
    Face.LEFT: 'orange',
    Face.RIGHT: 'red'
}

FACE_IDS: Dict[Face, int] = {
    Face.FRONT: 0,
    Face.BACK: 1,
    Face.UP: 2,
    Face.DOWN: 3,
    Face.LEFT: 4,
    Face.RIGHT: 5
}

OPPOSITE_FACES: Dict[Face, Face] = {
    Face.FRONT: Face.BACK,
    Face.BACK: Face.FRONT,
    Face.UP: Face.DOWN,
    Face.DOWN: Face.UP,
    Face.LEFT: Face.RIGHT,
    Face.RIGHT: Face.LEFT
}


# Plots
FACE_AXS: Dict[Face, tuple] = {
    Face.FRONT: (1, 1),
    Face.BACK: (1, 3),
    Face.UP: (0, 1),
    Face.DOWN: (2, 1),
    Face.LEFT: (1, 0),
    Face.RIGHT: (1, 2)
}

# Create a colormap for the faces
colors_list: List[str] = [''] * 6
for face, color in FACE_COLORS.items():
    colors_list[FACE_IDS[face]] = color
face_cmap = ListedColormap(colors_list)


class Cube:
    def __init__(self, cube: Optional[np.ndarray] = None, n_scramble_moves: int = 0, scramble_seed: Optional[int] = None) -> None:
        """Initialize the cube. If a cube is provided, it will be used; otherwise, a solved cube is created."""
        if cube is not None:
            self._cube = cube.copy()
        else:
            self._cube: np.ndarray = np.zeros((6, 3, 3), dtype=np.uint8)
            self.reset()
            self.scramble(n_scramble_moves, scramble_seed)

    def __str__(self) -> str:
        return '\n'.join(
            f'{face.name}:\n{self._cube[face_id]}' for face, face_id in FACE_IDS.items()
        )

    def __repr__(self) -> str:
        cube_repr = ''
        for face_id in self._cube.flatten():
            cube_repr += colors_list[face_id][0]
        return cube_repr

    def __hash__(self) -> int:
        """Return a hash of the cube state."""
        return hash(self.__repr__())

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Cube):
            return NotImplemented
        return np.array_equal(self._cube, other._cube)

    def move(self, move: Move | List[Move]) -> None:
        """Perform a move or sequence of moves on the cube."""
        if isinstance(move, list):
            for m in move:
                self.move(m)
            return

        n = 1
        if move.specifier == MoveSpecifier.COUNTERCLOCKWISE:
            n = 3
        elif move.specifier == MoveSpecifier.HALF_TURN:
            n = 2

        # Rotate the face
        for _ in range(n):
            self._rotate_face(move.face)

    def reset(self) -> None:
        """Reset the cube to its initial state."""
        self._cube.fill(0)
        for face_id in range(6):
            self._cube[face_id, :, :] = face_id

    @staticmethod
    def generate_scramble_sequence(n_moves: int, seed: Optional[int] = None) -> List[Move]:
        """Generate a scramble sequence of n_moves."""
        rng = np.random.default_rng(seed)
        faces: List[Face] = list(Face)
        moves = [Move(faces[rng.integers(0, len(faces))], MoveSpecifier(
            rng.integers(1, 4))) for _ in range(n_moves)]

        # Avoid trivial non-scrambling sequences:
        i = 0
        while i < len(moves) - 1:
            # 1. remove consecutive moves on the same face
            if moves[i].face == moves[i + 1].face:
                moves.pop(i + 1)
                moves.append(
                    Move(faces[rng.integers(0, len(faces))], MoveSpecifier(rng.integers(1, 4))))
                i = max(0, i - 1)
                continue

            # 2. ABA with A and B being opposite faces
            if i < len(moves) - 2 and moves[i].face == moves[i + 2].face and moves[i].face == OPPOSITE_FACES[moves[i + 1].face]:
                moves.pop(i + 2)
                moves.append(
                    Move(faces[rng.integers(0, len(faces))], MoveSpecifier(rng.integers(1, 4))))
                i -= 1
            i += 1

        return moves

    def scramble(self, n_moves: int, seed: Optional[int] = None) -> List[Move]:
        """Scramble the cube with a list of moves or a number of random moves. \n
        Returns the list of moves performed.
        """
        moves = Cube.generate_scramble_sequence(n_moves, seed)
        self.move(moves)
        return moves

    def is_solved(self) -> bool:
        """Check if the cube is solved."""
        for face_id in range(6):
            if not np.all(self._cube[face_id] == face_id):
                return False
        return True

    def copy(self) -> 'Cube':
        """Return a copy of the cube."""
        return Cube(self._cube.copy())

    def _rotate_face(self, face: Face) -> None:
        """Rotate a face 90 degrees clockwise."""
        self._rotate_face_surface(face)
        self._rotate_adjacent_faces(face)

    def _rotate_face_surface(self, face: Face) -> None:
        """Rotate the surface of the face 90 degrees clockwise."""
        face_id = FACE_IDS[face]

        # Rotate corners
        c0 = self._cube[face_id, 0, 0]
        self._cube[face_id, 0, 0] = self._cube[face_id, 2, 0]
        self._cube[face_id, 2, 0] = self._cube[face_id, 2, 2]
        self._cube[face_id, 2, 2] = self._cube[face_id, 0, 2]
        self._cube[face_id, 0, 2] = c0

        # Rotate edges
        e0 = self._cube[face_id, 0, 1]
        self._cube[face_id, 0, 1] = self._cube[face_id, 1, 0]
        self._cube[face_id, 1, 0] = self._cube[face_id, 2, 1]
        self._cube[face_id, 2, 1] = self._cube[face_id, 1, 2]
        self._cube[face_id, 1, 2] = e0

    def _rotate_adjacent_faces(self, face: Face) -> None:
        """Rotate the adjacent faces when a face is rotated."""

        match face:
            case Face.FRONT:
                # Rotate corners
                c0 = self._cube[FACE_IDS[Face.UP], 2, 0]
                self._cube[FACE_IDS[Face.UP], 2,
                           0] = self._cube[FACE_IDS[Face.LEFT], 2, 2]
                self._cube[FACE_IDS[Face.LEFT], 2,
                           2] = self._cube[FACE_IDS[Face.DOWN], 0, 2]
                self._cube[FACE_IDS[Face.DOWN], 0,
                           2] = self._cube[FACE_IDS[Face.RIGHT], 0, 0]
                self._cube[FACE_IDS[Face.RIGHT], 0, 0] = c0

                c1 = self._cube[FACE_IDS[Face.UP], 2, 2]
                self._cube[FACE_IDS[Face.UP], 2,
                           2] = self._cube[FACE_IDS[Face.LEFT], 0, 2]
                self._cube[FACE_IDS[Face.LEFT], 0,
                           2] = self._cube[FACE_IDS[Face.DOWN], 0, 0]
                self._cube[FACE_IDS[Face.DOWN], 0,
                           0] = self._cube[FACE_IDS[Face.RIGHT], 2, 0]
                self._cube[FACE_IDS[Face.RIGHT], 2, 0] = c1

                # Rotate edges
                e0 = self._cube[FACE_IDS[Face.UP], 2, 1]
                self._cube[FACE_IDS[Face.UP], 2,
                           1] = self._cube[FACE_IDS[Face.LEFT], 1, 2]
                self._cube[FACE_IDS[Face.LEFT], 1,
                           2] = self._cube[FACE_IDS[Face.DOWN], 0, 1]
                self._cube[FACE_IDS[Face.DOWN], 0,
                           1] = self._cube[FACE_IDS[Face.RIGHT], 1, 0]
                self._cube[FACE_IDS[Face.RIGHT], 1, 0] = e0

            case Face.BACK:
                # Rotate corners
                c0 = self._cube[FACE_IDS[Face.UP], 0, 2]
                self._cube[FACE_IDS[Face.UP], 0,
                           2] = self._cube[FACE_IDS[Face.RIGHT], 2, 2]
                self._cube[FACE_IDS[Face.RIGHT], 2,
                           2] = self._cube[FACE_IDS[Face.DOWN], 2, 0]
                self._cube[FACE_IDS[Face.DOWN], 2,
                           0] = self._cube[FACE_IDS[Face.LEFT], 0, 0]
                self._cube[FACE_IDS[Face.LEFT], 0, 0] = c0

                c1 = self._cube[FACE_IDS[Face.UP], 0, 0]
                self._cube[FACE_IDS[Face.UP], 0,
                           0] = self._cube[FACE_IDS[Face.RIGHT], 0, 2]
                self._cube[FACE_IDS[Face.RIGHT], 0,
                           2] = self._cube[FACE_IDS[Face.DOWN], 2, 2]
                self._cube[FACE_IDS[Face.DOWN], 2,
                           2] = self._cube[FACE_IDS[Face.LEFT], 2, 0]
                self._cube[FACE_IDS[Face.LEFT], 2, 0] = c1

                # Rotate edges
                e0 = self._cube[FACE_IDS[Face.UP], 0, 1]
                self._cube[FACE_IDS[Face.UP], 0,
                           1] = self._cube[FACE_IDS[Face.RIGHT], 1, 2]
                self._cube[FACE_IDS[Face.RIGHT], 1,
                           2] = self._cube[FACE_IDS[Face.DOWN], 2, 1]
                self._cube[FACE_IDS[Face.DOWN], 2,
                           1] = self._cube[FACE_IDS[Face.LEFT], 1, 0]
                self._cube[FACE_IDS[Face.LEFT], 1, 0] = e0

            case Face.UP:
                # Rotate corners
                c0 = self._cube[FACE_IDS[Face.BACK], 0, 2]
                self._cube[FACE_IDS[Face.BACK], 0,
                           2] = self._cube[FACE_IDS[Face.LEFT], 0, 2]
                self._cube[FACE_IDS[Face.LEFT], 0,
                           2] = self._cube[FACE_IDS[Face.FRONT], 0, 2]
                self._cube[FACE_IDS[Face.FRONT], 0,
                           2] = self._cube[FACE_IDS[Face.RIGHT], 0, 2]
                self._cube[FACE_IDS[Face.RIGHT], 0, 2] = c0

                c1 = self._cube[FACE_IDS[Face.BACK], 0, 0]
                self._cube[FACE_IDS[Face.BACK], 0,
                           0] = self._cube[FACE_IDS[Face.LEFT], 0, 0]
                self._cube[FACE_IDS[Face.LEFT], 0,
                           0] = self._cube[FACE_IDS[Face.FRONT], 0, 0]
                self._cube[FACE_IDS[Face.FRONT], 0,
                           0] = self._cube[FACE_IDS[Face.RIGHT], 0, 0]
                self._cube[FACE_IDS[Face.RIGHT], 0, 0] = c1

                # Rotate edges
                e0 = self._cube[FACE_IDS[Face.BACK], 0, 1]
                self._cube[FACE_IDS[Face.BACK], 0,
                           1] = self._cube[FACE_IDS[Face.LEFT], 0, 1]
                self._cube[FACE_IDS[Face.LEFT], 0,
                           1] = self._cube[FACE_IDS[Face.FRONT], 0, 1]
                self._cube[FACE_IDS[Face.FRONT], 0,
                           1] = self._cube[FACE_IDS[Face.RIGHT], 0, 1]
                self._cube[FACE_IDS[Face.RIGHT], 0, 1] = e0

            case Face.DOWN:
                # Rotate corners
                c0 = self._cube[FACE_IDS[Face.FRONT], 2, 0]
                self._cube[FACE_IDS[Face.FRONT], 2,
                           0] = self._cube[FACE_IDS[Face.LEFT], 2, 0]
                self._cube[FACE_IDS[Face.LEFT], 2,
                           0] = self._cube[FACE_IDS[Face.BACK], 2, 0]
                self._cube[FACE_IDS[Face.BACK], 2,
                           0] = self._cube[FACE_IDS[Face.RIGHT], 2, 0]
                self._cube[FACE_IDS[Face.RIGHT], 2, 0] = c0

                c1 = self._cube[FACE_IDS[Face.FRONT], 2, 2]
                self._cube[FACE_IDS[Face.FRONT], 2,
                           2] = self._cube[FACE_IDS[Face.LEFT], 2, 2]
                self._cube[FACE_IDS[Face.LEFT], 2,
                           2] = self._cube[FACE_IDS[Face.BACK], 2, 2]
                self._cube[FACE_IDS[Face.BACK], 2,
                           2] = self._cube[FACE_IDS[Face.RIGHT], 2, 2]
                self._cube[FACE_IDS[Face.RIGHT], 2, 2] = c1

                # Rotate edges
                e0 = self._cube[FACE_IDS[Face.FRONT], 2, 1]
                self._cube[FACE_IDS[Face.FRONT], 2,
                           1] = self._cube[FACE_IDS[Face.LEFT], 2, 1]
                self._cube[FACE_IDS[Face.LEFT], 2,
                           1] = self._cube[FACE_IDS[Face.BACK], 2, 1]
                self._cube[FACE_IDS[Face.BACK], 2,
                           1] = self._cube[FACE_IDS[Face.RIGHT], 2, 1]
                self._cube[FACE_IDS[Face.RIGHT], 2, 1] = e0

            case Face.LEFT:
                # Rotate corners
                c0 = self._cube[FACE_IDS[Face.UP], 0, 0]
                self._cube[FACE_IDS[Face.UP], 0,
                           0] = self._cube[FACE_IDS[Face.BACK], 2, 2]
                self._cube[FACE_IDS[Face.BACK], 2,
                           2] = self._cube[FACE_IDS[Face.DOWN], 0, 0]
                self._cube[FACE_IDS[Face.DOWN], 0,
                           0] = self._cube[FACE_IDS[Face.FRONT], 0, 0]
                self._cube[FACE_IDS[Face.FRONT], 0, 0] = c0

                c1 = self._cube[FACE_IDS[Face.UP], 2, 0]
                self._cube[FACE_IDS[Face.UP], 2,
                           0] = self._cube[FACE_IDS[Face.BACK], 0, 2]
                self._cube[FACE_IDS[Face.BACK], 0,
                           2] = self._cube[FACE_IDS[Face.DOWN], 2, 0]
                self._cube[FACE_IDS[Face.DOWN], 2,
                           0] = self._cube[FACE_IDS[Face.FRONT], 2, 0]
                self._cube[FACE_IDS[Face.FRONT], 2, 0] = c1

                # Rotate edges
                e0 = self._cube[FACE_IDS[Face.UP], 1, 0]
                self._cube[FACE_IDS[Face.UP], 1,
                           0] = self._cube[FACE_IDS[Face.BACK], 1, 2]
                self._cube[FACE_IDS[Face.BACK], 1,
                           2] = self._cube[FACE_IDS[Face.DOWN], 1, 0]
                self._cube[FACE_IDS[Face.DOWN], 1,
                           0] = self._cube[FACE_IDS[Face.FRONT], 1, 0]
                self._cube[FACE_IDS[Face.FRONT], 1, 0] = e0

            case Face.RIGHT:
                # Rotate corners
                c0 = self._cube[FACE_IDS[Face.UP], 2, 2]
                self._cube[FACE_IDS[Face.UP], 2,
                           2] = self._cube[FACE_IDS[Face.FRONT], 2, 2]
                self._cube[FACE_IDS[Face.FRONT], 2,
                           2] = self._cube[FACE_IDS[Face.DOWN], 2, 2]
                self._cube[FACE_IDS[Face.DOWN], 2,
                           2] = self._cube[FACE_IDS[Face.BACK], 0, 0]
                self._cube[FACE_IDS[Face.BACK], 0, 0] = c0

                c1 = self._cube[FACE_IDS[Face.UP], 0, 2]
                self._cube[FACE_IDS[Face.UP], 0,
                           2] = self._cube[FACE_IDS[Face.FRONT], 0, 2]
                self._cube[FACE_IDS[Face.FRONT], 0,
                           2] = self._cube[FACE_IDS[Face.DOWN], 0, 2]
                self._cube[FACE_IDS[Face.DOWN], 0,
                           2] = self._cube[FACE_IDS[Face.BACK], 2, 0]
                self._cube[FACE_IDS[Face.BACK], 2, 0] = c1

                # Rotate edges
                e0 = self._cube[FACE_IDS[Face.UP], 1, 2]
                self._cube[FACE_IDS[Face.UP], 1,
                           2] = self._cube[FACE_IDS[Face.FRONT], 1, 2]
                self._cube[FACE_IDS[Face.FRONT], 1,
                           2] = self._cube[FACE_IDS[Face.DOWN], 1, 2]
                self._cube[FACE_IDS[Face.DOWN], 1,
                           2] = self._cube[FACE_IDS[Face.BACK], 1, 0]
                self._cube[FACE_IDS[Face.BACK], 1, 0] = e0

    def _rotate_cube_front(self) -> None:
        """Rotate the entire cube around the front face clockwise."""
        self._rotate_face_surface(Face.FRONT)
        for _ in range(3):
            self._rotate_face_surface(Face.BACK)
        f = self._cube[FACE_IDS[Face.UP], :, :].copy()
        self._cube[FACE_IDS[Face.UP], :,
                   :] = self._cube[FACE_IDS[Face.LEFT], :, :]
        self._rotate_face_surface(Face.UP)
        self._cube[FACE_IDS[Face.LEFT], :,
                   :] = self._cube[FACE_IDS[Face.DOWN], :, :]
        self._rotate_face_surface(Face.LEFT)
        self._cube[FACE_IDS[Face.DOWN], :,
                   :] = self._cube[FACE_IDS[Face.RIGHT], :, :]
        self._rotate_face_surface(Face.DOWN)
        self._cube[FACE_IDS[Face.RIGHT], :, :] = f
        self._rotate_face_surface(Face.RIGHT)

    def _rotate_cube_up(self) -> None:
        """Rotate the entire cube around the up face clockwise."""
        self._rotate_face_surface(Face.UP)
        for _ in range(3):
            self._rotate_face_surface(Face.DOWN)

        f = self._cube[FACE_IDS[Face.FRONT], :, :].copy()
        self._cube[FACE_IDS[Face.FRONT], :,
                   :] = self._cube[FACE_IDS[Face.RIGHT], :, :]
        self._cube[FACE_IDS[Face.RIGHT], :,
                   :] = self._cube[FACE_IDS[Face.BACK], :, :]
        self._cube[FACE_IDS[Face.BACK], :,
                   :] = self._cube[FACE_IDS[Face.LEFT], :, :]
        self._cube[FACE_IDS[Face.LEFT], :, :] = f

    def _rotate_cube_left(self) -> None:
        """Rotate the entire cube around the left face clockwise."""
        self._rotate_face_surface(Face.LEFT)
        for _ in range(3):
            self._rotate_face_surface(Face.RIGHT)

        f = self._cube[FACE_IDS[Face.FRONT], :, :].copy()
        self._cube[FACE_IDS[Face.FRONT], :,
                   :] = self._cube[FACE_IDS[Face.UP], :, :]
        self._cube[FACE_IDS[Face.UP], :,
                   :] = self._cube[FACE_IDS[Face.BACK], ::-1, ::-1]
        self._cube[FACE_IDS[Face.BACK], :,
                   :] = self._cube[FACE_IDS[Face.DOWN], ::-1, ::-1]
        self._cube[FACE_IDS[Face.DOWN], :, :] = f

    def _reset_colors(self) -> None:
        """Changes the colors of the cube to match the default face colors."""
        self._cube += 6  # Temporary offset to avoid overwriting
        for face_id in range(6):
            wrong_color = self._cube[face_id, 1, 1]
            self._cube[self._cube == wrong_color] = face_id

    def get_all_similar_cubes(self) -> List['Cube']:
        """Get all cubes that are similar to this one by rotating the entire cube."""
        front_cubes: List[Cube] = []  # Store one cube with each face in front

        front_cubes.append(self.copy())  # FRONT face in front

        up_cube = self.copy()
        up_cube._rotate_cube_left()  # Bring UP face to front
        front_cubes.append(up_cube)

        down_cube = self.copy()
        for _ in range(3):
            down_cube._rotate_cube_left()  # Bring DOWN face to front
        front_cubes.append(down_cube)

        right_cube = self.copy()
        right_cube._rotate_cube_up()  # Bring RIGHT face to front
        front_cubes.append(right_cube)

        back_cube = self.copy()
        for _ in range(2):
            back_cube._rotate_cube_up()  # Bring BACK face to front
        front_cubes.append(back_cube)

        left_cube = self.copy()
        for _ in range(3):
            left_cube._rotate_cube_up()  # Bring LEFT face to front
        front_cubes.append(left_cube)

        similar_cubes = []
        for i in range(4):
            for cube in front_cubes:
                new_cube = cube.copy()
                for _ in range(i):
                    new_cube._rotate_cube_front()
                new_cube._reset_colors()
                similar_cubes.append(new_cube)
        return similar_cubes

    def get_all_neighbors(self) -> List['Cube']:
        """Get all cubes that are one move away from this one."""
        neighbors = []
        for move in moves:
            new_cube = self.copy()
            new_cube.move(move)
            neighbors.append(new_cube)
        return neighbors

    def plot(self) -> None:
        """Plot the cube using matplotlib."""
        fig, axs = plt.subplots(3, 4)

        for face, face_id in FACE_IDS.items():
            face_ax = axs[FACE_AXS[face][0], FACE_AXS[face][1]]
            face_ax.imshow(self._cube[face_id], cmap=face_cmap, vmin=0, vmax=5)
            face_ax.set_title(face.name)

            # Draw grid lines
            for x in range(4):
                face_ax.axhline(x - 0.5, color='black', linewidth=2)
                face_ax.axvline(x - 0.5, color='black', linewidth=2)

        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def _draw_face(self, ax, face_id, face_center, u, v) -> None:
        """Draw a face with unit vectors u, v."""

        for i in range(3):
            for j in range(3):
                square_center = face_center + u * (j-1) + v * (i-1)
                verts = [
                    square_center + 0.5 * u + 0.5 * v,
                    square_center + 0.5 * u - 0.5 * v,
                    square_center - 0.5 * u - 0.5 * v,
                    square_center - 0.5 * u + 0.5 * v
                ]
                face_color = colors_list[self._cube[face_id, i, j]]
                poly = Poly3DCollection(
                    [verts], facecolors=face_color, edgecolors="black", linewidths=1)
                ax.add_collection3d(poly)

    def plot_3d(self, elev: float = 30, azim: float = 45, ax=None, show: bool = True):
        """Plot the cube in 3D.

        Args:
            elev: Elevation angle in degrees (default: 30)
            azim: Azimuthal angle in degrees (default: 45)
            ax: Optional matplotlib 3D axis to draw on. If None, creates new figure.
            show: Whether to call plt.show() at the end (default: True)
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        self._draw_face(ax, FACE_IDS[Face.FRONT], np.array(
            [1.5, 0, 0]),   np.array([0, 1, 0]),  np.array([0, 0, -1]))
        self._draw_face(ax, FACE_IDS[Face.BACK],  np.array(
            [-1.5, 0, 0]),  np.array([0, -1, 0]), np.array([0, 0, -1]))
        self._draw_face(ax, FACE_IDS[Face.UP],    np.array(
            [0, 0, 1.5]),   np.array([0, 1, 0]),  np.array([1, 0, 0]))
        self._draw_face(ax, FACE_IDS[Face.DOWN],  np.array(
            [0, 0, -1.5]), np.array([0, 1, 0]),  np.array([-1, 0, 0]))
        self._draw_face(ax, FACE_IDS[Face.LEFT],  np.array(
            [0, -1.5, 0]), np.array([1, 0, 0]),  np.array([0, 0, -1]))
        self._draw_face(ax, FACE_IDS[Face.RIGHT], np.array(
            [0, 1.5, 0]),   np.array([-1, 0, 0]), np.array([0, 0, -1]))

        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        ax.set_aspect('equal')

        if show:
            plt.show()

        return ax

    def plot_2d(self) -> None:
        """Plot the cube in 2D with two viewpoints (front and back)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
            12, 6), subplot_kw={'projection': '3d'})

        # Front view
        self.plot_3d(elev=30, azim=45, ax=ax1, show=False)
        ax1.set_title('Front View', fontsize=14, fontweight='bold')

        # Back view
        self.plot_3d(elev=30, azim=225, ax=ax2, show=False)
        ax2.set_title('Back View', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()


def main():
    c = Cube()
    c.scramble(20)
    print(c)
    c.plot_3d()


if __name__ == "__main__":
    main()
