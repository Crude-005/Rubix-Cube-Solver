import random
import queue
import numpy as np
import time


N = 3

# Step1: Data structure to store Cube
# 3x3 x6 arr
#         5 5 5
#         5 5 5
#         5 5 5
#
# 3 3 3   1 1 1   4 4 4
# 3 3 3   1 1 1   4 4 4
# 3 3 3   1 1 1   4 4 4
#
#         2 2 2
#         2 2 2
#         2 2 2
#
#         6 6 6
#         6 6 6
#         6 6 6


# 1 is front
# 2 is bottom
# 3 is left
# 4 is right
# 5 is up
# 6 is back


def showCube(cube):
    for i in range(3):
        print("\t", end="")
        print(*cube[4][i], sep=" ")
    print()
    for i in range(3):
        print(*cube[2][i], sep=" ", end="\t")
        print(*cube[0][i], sep=" ", end="\t")
        print(*cube[3][i], sep=" ", end="\n")
    print()
    for i in range(3):
        print("\t", end="")
        print(*cube[1][i], sep=" ")
    print()
    for i in range(3):
        print("\t", end="")
        print(*cube[5][i], sep=" ")
    print()


def getFinalState(n):
    finalState = np.arange(6 * n * n, dtype=np.int64).reshape(6, n, n)
    for i in range(1, 7):
        finalState[i - 1][::] = i
    return finalState


def generateValidMoves(n):
    arr = list()
    for i in range(1, 7):
        for j in range(1, n // 2 + 1):
            for k in range(1, 4):
                arr.append([i, j, k])
    return arr


def initialiseCube(n):
    select = input(
        "Enter 0 to use default Cube or 1 to input Custom Cube or 2 for random Cube : "
    )
    if select == "2":
        validMoves = generateValidMoves(n)
        movesToRandomise = 5
        moves = [random.choice(validMoves) for _ in range(movesToRandomise)]
        print("Moves")
        print(*moves, sep="\n")
        print()
        startState = getFinalState(n)
        for move in moves:
            startState = playMove(move, startState)
        return startState

    elif select == "1":
        startState = list()
        face = list()
        print("Front Face:\n ")
        for i in range(n):
            face.append(list(map(int, input(f"Enter layer {i+1}: ").split())))
        startState.append(face)
        face = list()
        print("Bottom Face:\n ")
        for i in range(n):
            face.append(list(map(int, input(f"Enter layer {i+1}: ").split())))
        startState.append(face)
        face = list()
        print("Left Face:\n ")
        for i in range(n):
            face.append(list(map(int, input(f"Enter layer {i+1}: ").split())))
        startState.append(face)
        face = list()
        print("Right Face:\n ")
        for i in range(n):
            face.append(list(map(int, input(f"Enter layer {i+1}: ").split())))
        startState.append(face)
        face = list()
        print("Up Face:\n ")
        for i in range(n):
            face.append(list(map(int, input(f"Enter layer {i+1}: ").split())))
        startState.append(face)
        face = list()
        print("Back Face:\n ")
        for i in range(n):
            face.append(list(map(int, input(f"Enter layer {i+1}: ").split())))
        startState.append(face)

    else:
        if n == 3:
            startState = [
                [[4, 6, 6], [2, 1, 2], [2, 4, 1]],
                [[6, 2, 4], [1, 2, 5], [3, 6, 3]],
                [[3, 6, 2], [3, 3, 3], [2, 4, 3]],
                [[5, 6, 2], [1, 4, 3], [5, 4, 6]],
                [[1, 1, 4], [3, 5, 2], [1, 5, 4]],
                [[1, 4, 5], [1, 6, 5], [5, 5, 6]],
            ]
        else:
            startState = getFinalState(n)

    return np.array(startState)


def playMove(move, previousState):
    currentState = np.copy(previousState)
    n = len(currentState[0])

    if move[0] == 1:
        if move[1] == 1:
            currentState[0] = np.rot90(currentState[0], k=-move[2])
        for _ in range(move[2]):
            for i in range(n):
                (
                    currentState[1][move[1] - 1][i],
                    currentState[2][i][n - move[1]],
                    currentState[4][n - move[1]][n - i - 1],
                    currentState[3][n - i - 1][move[1] - 1],
                ) = (
                    currentState[3][n - i - 1][move[1] - 1],
                    currentState[1][move[1] - 1][i],
                    currentState[2][i][n - move[1]],
                    currentState[4][n - move[1]][n - i - 1],
                )
    elif move[0] == 6:
        if move[1] == 1:
            currentState[5] = np.rot90(currentState[5], k=move[2])
        for _ in range(move[2]):
            for i in range(n):
                (
                    currentState[1][n - move[1]][i],
                    currentState[2][i][move[1] - 1],
                    currentState[4][move[1] - 1][n - i - 1],
                    currentState[3][n - i - 1][n - move[1]],
                ) = (
                    currentState[3][n - i - 1][n - move[1]],
                    currentState[1][n - move[1]][i],
                    currentState[2][i][move[1] - 1],
                    currentState[4][move[1] - 1][n - i - 1],
                )
    elif move[0] == 2:
        if move[1] == 1:
            currentState[1] = np.rot90(currentState[1], k=-move[2])
        for _ in range(move[2]):
            for i in range(n):
                (
                    currentState[3][n - move[1]][i],
                    currentState[0][n - move[1]][i],
                    currentState[2][n - move[1]][i],
                    currentState[5][move[1] - 1][n - i - 1],
                ) = (
                    currentState[0][n - move[1]][i],
                    currentState[2][n - move[1]][i],
                    currentState[5][move[1] - 1][n - i - 1],
                    currentState[3][n - move[1]][i],
                )
    elif move[0] == 5:
        if move[1] == 1:
            currentState[4] = np.rot90(currentState[4], k=move[2])
        for _ in range(move[2]):
            for i in range(n):
                (
                    currentState[5][n - move[1]][n - i - 1],
                    currentState[2][move[1] - 1][i],
                    currentState[0][move[1] - 1][i],
                    currentState[3][move[1] - 1][i],
                ) = (
                    currentState[3][move[1] - 1][i],
                    currentState[5][n - move[1]][n - i - 1],
                    currentState[2][move[1] - 1][i],
                    currentState[0][move[1] - 1][i],
                )
    elif move[0] == 4:
        if move[1] == 1:
            currentState[3] = np.rot90(currentState[3], k=-move[2])
        for _ in range(move[2]):
            for i in range(n):
                (
                    currentState[5][i][n - move[1]],
                    currentState[1][i][n - move[1]],
                    currentState[0][i][n - move[1]],
                    currentState[4][i][n - move[1]],
                ) = (
                    currentState[4][i][n - move[1]],
                    currentState[5][i][n - move[1]],
                    currentState[1][i][n - move[1]],
                    currentState[0][i][n - move[1]],
                )
    elif move[0] == 3:
        if move[1] == 1:
            currentState[2] = np.rot90(currentState[2], k=move[2])
        for _ in range(move[2]):
            for i in range(n):
                (
                    currentState[5][i][move[1] - 1],
                    currentState[1][i][move[1] - 1],
                    currentState[0][i][move[1] - 1],
                    currentState[4][i][move[1] - 1],
                ) = (
                    currentState[4][i][move[1] - 1],
                    currentState[5][i][move[1] - 1],
                    currentState[1][i][move[1] - 1],
                    currentState[0][i][move[1] - 1],
                )
    else:
        print("\nWarning" * 10)
        print("Brother I doubt your moves suggestion Functions is working right")

    return currentState


def cubeToString(cube):
    s = ""
    for i in cube:
        for j in i:
            for k in j:
                s += str(k)
    return s


def stringToCube(s):
    if s == "":
        return None
    n = int((len(s) / 6) ** (1 / 2))

    cnt = 0
    cube = list()
    for i in range(6):
        face = list()
        for j in range(n):
            layer = list()
            for k in range(n):
                layer.append(int(s[cnt]))
                cnt += 1
            face.append(layer)
        cube.append(face)
    return np.array(cube)



# ==== CHANGED: improved heuristic ====

def _st(state, face, r, c):
    return int(state[face][r][c])


def _corner_positions(n):
    return [
        [(4, n - 1, 0), (2, 0, n - 1), (0, 0, 0)],
        [(4, n - 1, n - 1), (0, 0, n - 1), (3, 0, 0)],
        [(4, 0, 0), (3, 0, n - 1), (5, 0, n - 1)],
        [(4, 0, n - 1), (5, 0, 0), (3, 0, n - 1)],
        [(1, 0, 0), (0, n - 1, 0), (2, n - 1, n - 1)],
        [(1, 0, n - 1), (3, n - 1, 0), (0, n - 1, n - 1)],
        [(1, n - 1, 0), (5, n - 1, n - 1), (2, n - 1, 0)],
        [(1, n - 1, n - 1), (2, n - 1, 0), (5, n - 1, 0)],
    ]


def _edge_positions(n):
    return [
        [(4, n - 1, 1), (0, 0, 1)],
        [(4, 1 if n == 3 else n - 1, n - 1), (3, 0, 1)],
        [(4, 1 if n == 3 else n - 1, 0), (2, 0, 1)],
        [(4, 0, 1), (5, 0, 1)],
        [(1, 0, 1), (0, n - 1, 1)],
        [(1, 0, n - 2 if n > 3 else n - 1), (3, n - 1, 1)],
        [(1, 0, 1), (2, n - 1, 1)],
        [(1, n - 1, 1), (5, n - 1, 1)],
        [(2, 1 if n == 3 else n - 1, 0), (0, 1 if n == 3 else 0, 0)],
        [(3, 1 if n == 3 else n - 1, 0), (0, 1 if n == 3 else 0, n - 1)],
        [(2, 1 if n == 3 else 0, n - 1), (5, 1 if n == 3 else 0, n - 1)],
        [(3, 1 if n == 3 else 0, n - 1), (5, 1 if n == 3 else 0, 0)],
    ]


def h_corner_orientation(state, goal):
    n = len(state[0])
    corners = _corner_positions(n)
    mis_twist = 0
    for corner in corners:
        goal_faces = set(f + 1 for (f, _, _) in corner)
        colors = [_st(state, f, r, c) for (f, r, c) in corner]

        if set(colors) == goal_faces:
            if 5 in goal_faces or 2 in goal_faces:
                ud = 5 if 5 in goal_faces else 2
                pos_faces = [f + 1 for (f, _, _) in corner]
                try:
                    ci = colors.index(ud)
                    if pos_faces[ci] != ud:
                        mis_twist += 1
                except:
                    mis_twist += 1
    return int(np.ceil(mis_twist / 2))


def h_edge_orientation(state, goal):
    n = len(state[0])
    edges = _edge_positions(n)
    mis_flip = 0
    for edge in edges:
        goal_faces = set(f + 1 for (f, _, _) in edge)
        colors = [_st(state, f, r, c) for (f, r, c) in edge]

        if set(colors) == goal_faces:
            ud = None
            if 5 in goal_faces:
                ud = 5
            elif 2 in goal_faces:
                ud = 2

            if ud is not None:
                pos_faces = [f + 1 for (f, _, _) in edge]
                try:
                    ci = colors.index(ud)
                    if pos_faces[ci] != ud:
                        mis_flip += 1
                except:
                    mis_flip += 1
            else:
                pos_faces = [f + 1 for (f, _, _) in edge]
                if not any(colors[i] == pos_faces[i] for i in range(2)):
                    mis_flip += 1

    return int(np.ceil(mis_flip / 2))


def h_cubie_manhattan(state, goal):
    n = len(state[0])
    corners = _corner_positions(n)
    edges = _edge_positions(n)
    mis = 0

    for corner in corners:
        goal_faces = set(f + 1 for (f, _, _) in corner)
        colors = [_st(state, f, r, c) for (f, r, c) in corner]
        if set(colors) != goal_faces:
            mis += 1

    for edge in edges:
        goal_faces = set(f + 1 for (f, _, _) in edge)
        colors = [_st(state, f, r, c) for (f, r, c) in edge]
        if set(colors) != goal_faces:
            mis += 1

    return int(np.ceil(mis / 4))


def heuristic(state, goal):
    return max(
        h_corner_orientation(state, goal),
        h_edge_orientation(state, goal),
        h_cubie_manhattan(state, goal)
    )



# ==== CHANGED: updated search_ida to return moves ====
def search_ida(path, moves_taken, g, bound, goal, moves):
    node = path[-1]
    f = g + heuristic(node, goal)

    if f > bound:
        return f, None

    if np.array_equal(node, goal):
        return "FOUND", moves_taken.copy()

    min_bound = float("inf")

    for mv in moves:
        new_state = playMove(mv, node)

        skip = False
        for p in path:
            if np.array_equal(p, new_state):
                skip = True
                break
        if skip:
            continue

        path.append(new_state)
        moves_taken.append(mv)

        t, sol = search_ida(path, moves_taken, g + 1, bound, goal, moves)

        if t == "FOUND":
            return "FOUND", sol

        if t < min_bound:
            min_bound = t

        path.pop()
        moves_taken.pop()

    return min_bound, None



# ==== CHANGED: updated ida_star to return move list ====
def ida_star(start, goal, moves):
    bound = heuristic(start, goal)
    path = [start]
    moves_taken = []

    while True:
        t, sol = search_ida(path, moves_taken, 0, bound, goal, moves)

        if t == "FOUND":
            return sol

        if t == float("inf"):
            return None

        bound = t



# ==== CHANGED: solve now prints move sequence ====
def solve(startState: np.ndarray, finalState, n):

    showCube(startState)

    moves = generateValidMoves(n)

    result = ida_star(startState, finalState, moves)

    if result is None:
        print("Sorry buddy you messed up")
    else:
        print("Moves to solve:")
        print(result)
        print("Solution length:", len(result))



def main():
    finalState = getFinalState(N)
    startState = initialiseCube(N)
    start = time.time()

    solve(startState, finalState, N)
    end = time.time()
    print(f"Time Taken = {end - start:.3f} s.")


if __name__ == "__main__":
    main()
