import random
import queue


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
    finalState = list()
    for i in range(1, 7):
        face = list()
        for _ in range(n):
            face.append([i] * n)
        finalState.append(face)
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
    startState = list()
    if select == "2":
        validMoves = generateValidMoves(n)
        moves = [random.choice(validMoves) for _ in range(3)]
        print("Moves")
        print(*moves, sep="\n")
        print()
        startState = getFinalState(n)
        for move in moves:
            startState = playMove(move, startState)
        return startState

    elif select == "1":
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
            # =====================================
            startState = getFinalState(n)

        # =====================================
    return startState


def faceRotate(face, val, clockwise=True):
    def rotate_right(f):
        return [list(reversed(col)) for col in zip(*f)]

    def rotate_left(f):
        return [list(col) for col in zip(*f)][::-1]

    result = [row[:] for row in face]
    if clockwise:
        for _ in range(val):
            result = rotate_right(result)
    else:
        for _ in range(val):
            result = rotate_left(result)
    return result


def playMove(move, previousState):
    currentState = previousState.copy()
    n = len(currentState[0])

    if move[0] == 1:
        if move[1] == 1:
            currentState[0] = faceRotate(currentState[0], move[2])
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
            currentState[5] = faceRotate(currentState[5], move[2], False)
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
            currentState[1] = faceRotate(currentState[1], move[2])
        for _ in range(move[2]):
            (
                currentState[5][move[1] - 1],
                currentState[2][n - move[1]],
                currentState[0][n - move[1]],
                currentState[3][n - move[1]],
            ) = (
                currentState[3][n - move[1]][::-1],
                currentState[5][move[1] - 1][::-1],
                currentState[2][n - move[1]],
                currentState[0][n - move[1]],
            )
    elif move[0] == 5:
        if move[1] == 1:
            currentState[4] = faceRotate(currentState[4], move[2], False)
        for _ in range(move[2]):
            (
                currentState[5][n - move[1]],
                currentState[2][move[1] - 1],
                currentState[0][move[1] - 1],
                currentState[3][move[1] - 1],
            ) = (
                currentState[3][move[1] - 1][::-1],
                currentState[5][n - move[1]][::-1],
                currentState[2][move[1] - 1],
                currentState[0][move[1] - 1],
            )
    elif move[0] == 4:
        if move[1] == 1:
            currentState[3] = faceRotate(currentState[3], move[2])
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
            currentState[2] = faceRotate(currentState[2], move[2], False)
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
    n = (s / 6) ** 1 / 2
    cnt = 0
    cube = list()
    for i in 6:
        face = list()
        for j in range(n):
            layer = list()
            for k in range(n):
                layer.append(s[int(cnt)])
                cnt += 1
        face.append(layer)
    cube.append(face)
    return cube


def solve(startState, finalState, n):

    validMoves = generateValidMoves(n)
    parent = dict()
    parent[cubeToString(startState)] = [None, None]
    q = queue.Queue()
    q.put(startState)
    solutionFound = False

    while not q.empty() or solutionFound:
        currentState = q.get()
        for move in validMoves:
            nextState = playMove(move, currentState)
            if parent.get(cubeToString(nextState), -1) == -1:
                parent[cubeToString(nextState)] = (move, currentState)
                q.put(nextState)
            else:
                continue
            if nextState == finalState:
                solutionFound = True
                break
    if not solutionFound:
        print("Sorry buddy you messed up")
    else:
        backtrackArr = list()
        solutionMoves = list()
        backtrackArr.append(finalState)
        curr = finalState
        while curr:
            st = cubeToString(curr)
            solutionMoves.append(parent[st][0])
            backtrackArr.append(parent[st][1])
            curr = parent[st][1]
        print(*backtrackArr[::-1], sep="\n\n")
        print(solutionMoves[::-1])


def main():
    finalState = getFinalState(N)
    startState = initialiseCube(N)

    solve(startState, finalState, N)


if __name__ == "__main__":
    main()
