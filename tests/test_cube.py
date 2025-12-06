import pytest
import numpy as np
from cube import Cube, Move, Face, MoveSpecifier, FACE_COLORS, FACE_IDS, OPPOSITE_FACES
from matplotlib.colors import is_color_like


def test_face():
    assert len(Face) == 6


def test_move_specifier():
    assert len(MoveSpecifier) == 3


def test_face_colors():
    assert len(FACE_COLORS) == 6
    for face in list(Face):
        assert face in FACE_COLORS.keys()
        assert is_color_like(
            FACE_COLORS[face]), f"Color for {face} is not a valid color. Got: {FACE_COLORS[face]}"


def test_opposite_faces():
    assert len(OPPOSITE_FACES) == 6
    for face in list(Face):
        assert face in OPPOSITE_FACES.keys()
        opposite = OPPOSITE_FACES[face]
        assert opposite in list(
            Face), f"Opposite face for {face} is not a valid face. Got: {opposite}"
        assert OPPOSITE_FACES[
            opposite] == face, f"Opposite face mapping is not symmetric for {face} and {opposite}."


def test_face_ids():
    assert len(FACE_IDS) == 6
    for face in list(Face):
        assert face in FACE_IDS.keys()
        assert isinstance(FACE_IDS[face], int)
        assert 0 <= FACE_IDS[face] < 6, f"ID for {face} is out of range. Got: {FACE_IDS[face]}"
    for i in range(6):
        assert i in FACE_IDS.values(), f"ID {i} is not assigned to any face."


@pytest.fixture(params=list(Face))
def face(request):
    return request.param


@pytest.fixture(params=list(MoveSpecifier))
def specifier(request):
    return request.param


@pytest.fixture
def move(face: Face, specifier: MoveSpecifier):
    return Move(face, specifier)


def test_move(face: Face, specifier: MoveSpecifier):
    move = Move(face, specifier)
    assert move.face == face
    assert move.specifier == specifier

    inverse = move.get_inverse()
    assert inverse.face == face
    if specifier == MoveSpecifier.CLOCKWISE:
        assert inverse.specifier == MoveSpecifier.COUNTERCLOCKWISE
    elif specifier == MoveSpecifier.COUNTERCLOCKWISE:
        assert inverse.specifier == MoveSpecifier.CLOCKWISE
    elif specifier == MoveSpecifier.HALF_TURN:
        assert inverse.specifier == MoveSpecifier.HALF_TURN


def test_cube_initialization():
    cube = Cube()
    assert cube._cube.shape == (6, 3, 3)
    assert cube.is_solved()


@pytest.mark.parametrize("n_moves", [2, 7, 15, 25])
@pytest.mark.parametrize("seed", [1, 11, 111, 1111, 11111])
def test_cube_scramble(n_moves, seed):
    cube = Cube()
    moves = cube.scramble(n_moves, seed=seed)
    cube2 = Cube(n_scramble_moves=n_moves, scramble_seed=seed)
    check_cube_validity(cube)
    check_cube_validity(cube2)
    assert np.array_equal(
        cube._cube, cube2._cube), "Cubes scrambled with the same seed and number of moves should be identical."
    assert len(
        moves) == n_moves, "Scramble should return the correct number of moves."
    assert not cube.is_solved(), "Cube should not be solved after scrambling."
    for move in reversed(moves):
        cube.move(move.get_inverse())
        check_cube_validity(cube)
    assert cube.is_solved(
    ), "Cube should be solved after applying the inverse moves of the scramble."


def test_cube_scramble_patterns():
    cube = Cube()
    for _ in range(50):
        moves = cube.scramble(40, seed=54321)
        check_cube_validity(cube)
        assert len(moves) == 40
        for i in range(len(moves)-1):
            assert moves[i].face != moves[i +
                                          1].face, "Consecutive moves in the scramble should not be on the same face."

        for i in range(len(moves)-2):
            assert not (moves[i].face == moves[i+2].face and moves[i].face ==
                        OPPOSITE_FACES[moves[i+1].face]), "ABA patterns should be avoided in the scramble."


@pytest.fixture(params=[
    (1, 2),
    (2, 8),
    (3, 12),
    (4, 20),
    (5, 30)
])
def c(request):
    seed, moves = request.param
    cube = Cube()
    cube.scramble(moves, seed=seed)
    return cube


def test_cube_copy(c: Cube):
    copied_cube = c.copy()
    assert np.array_equal(
        c._cube, copied_cube._cube), "Copied cube should have the same state as the original."
    assert c is not copied_cube, "Copied cube should be a different instance."
    copied_cube.move(Move(Face.FRONT, MoveSpecifier.CLOCKWISE))
    assert not np.array_equal(
        c._cube, copied_cube._cube), "Cube state should change after a move on the copied cube."


def test_cube_move(c: Cube, move: Move):
    new_cube = c.copy()
    new_cube.move(move)
    assert not np.array_equal(
        c._cube, new_cube._cube), "Cube state should change after a move."
    check_cube_validity(new_cube)
    new_cube.move(move.get_inverse())
    assert np.array_equal(
        c._cube, new_cube._cube), "Cube state should return to original after applying inverse move."
    check_cube_validity(new_cube)


def test_similar_cubes(c: Cube):
    similar_cubes = c.get_all_similar_cubes()
    assert len(
        similar_cubes) == 24, "There should be exactly 24 similar cube configurations."

    similar_cubes_set = set(similar_cubes)
    for similar_cube in similar_cubes:
        check_cube_validity(similar_cube)
        similar_cubes_similar = similar_cube.get_all_similar_cubes()
        similar_cubes_similar_set = set(similar_cubes_similar)
        assert similar_cubes_set == similar_cubes_similar_set, "Similar cubes should be consistent across transformations."


def test_neighbors(c: Cube):
    neighbors = c.get_all_neighbors()
    assert len(
        neighbors) == 18, "There should be exactly 18 neighboring cube configurations."

    for neighbor in neighbors:
        check_cube_validity(neighbor)
        assert not np.array_equal(
            c._cube, neighbor._cube), "Neighboring cube state should be different from the original."


def check_cube_validity(c: Cube):
    state = c._cube.flatten()
    assert state.min() == 0
    assert state.max() == 5

    for i in range(6):
        assert np.sum(
            state == i) == 9, f"Face {i} should have exactly 9 stickers."
