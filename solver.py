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


# There are 6 face that we can rotate
# Front Back up bottom left and right
# each face has (n-1) / 2 layer that can be moved. 1 at edge to n/2 at centre.
# it can rotate by either 1 ,2 ,3.
# We want layer to move in one direction only
# Front bottom and right are moves in clockwise direction while back , up and right moves anti clock wise

# [1, 1, 1] -> means 1st layer front(1) will rotate 1 time
# [1, 1, 2] -> means 1st layer front(1) will rotate 2 time
# [1, 1, 3] -> means 1st layer front(1) will rotate 3 time
# [2, 1, 1] -> means 1st layer bottom(2) will rotate 1 time
# [2, 1, 2] -> means 1st layer bottom(2) will rotate 2 time
# [2, 1, 3]
# [3, 1, 1]
# [3, 1, 2]
# [3, 1, 3]
# [4, 1, 1]
# [4, 1, 2]
# [4, 1, 3]
# [5, 1, 1]
# [5, 1, 2]
# [5, 1, 3]
# [6, 1, 1]
# [6, 1, 2]
# [6, 1, 3]

# This is for cube with odd sides
# TO BE FIGURED OUT IN FUTURE


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
        movesToRandomise = 6
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
        # print(startState)
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
            # =====================================
            startState = getFinalState(n)

        # =====================================
    return np.array(startState)


# def faceRotate(face, val, clockwise=True):
#     print(face)

#     def rotate_right(f):
#         return np.array([np.array(reversed(col)) for col in zip(*f)])

#     def rotate_left(f):
#         return np.flip(np.array(np.array(col) for col in zip(*f)))

#     if clockwise:
#         for _ in range(val):
#             face = rotate_right(face)
#     else:
#         for _ in range(val):
#             face = rotate_left(face)
#     return face


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


def solve(startState: np.ndarray, finalState, n):

    validMoves = generateValidMoves(n)
    parent = dict()
    showCube(startState)

    finalStateString = cubeToString(finalState)
    parent[cubeToString(startState)] = [None, ""]
    q = queue.Queue()
    q.put(startState)
    solutionNotFound = True

    while not q.empty() and solutionNotFound:
        currentState = q.get()
        currentStateString = cubeToString(currentState)
        for move in validMoves:
            # showCube(currentState)
            # print(move)
            nextState = playMove(move, currentState)
            # showCube(nextState)
            if parent.get(cubeToString(nextState), -1) == -1:
                parent[cubeToString(nextState)] = (move, currentStateString)
                q.put(nextState)
            else:
                continue
            if cubeToString(nextState) == finalStateString:
                solutionNotFound = False
                break
    if solutionNotFound:
        print("Sorry buddy you messed up")
    else:
        backtrackArr = list()
        solutionMoves = list()
        backtrackArr.append(finalStateString)
        curr = finalStateString
        while curr != "":
            solutionMoves.append(parent[curr][0])
            backtrackArr.append(stringToCube(parent[curr][1]))
            curr = parent[curr][1]
        print(*backtrackArr[::-1], sep="\n\n")
        print(solutionMoves[::-1])


def main():
    finalState = getFinalState(N)
    startState = initialiseCube(N)
    start = time.time()

    solve(startState, finalState, N)
    end = time.time()
    print(f"Time Taken = {end - start:.3f} s.")


if __name__ == "__main__":
    main()
