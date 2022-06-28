#!/usr/bin/env python3
"""
Here we define the sudoku solver class, that deals with the core algorithm of solving a sudoku
"""
import time
from copy import deepcopy
import numpy as np


class SudokuSolver:
    """
    We pass the problem of the sudoku as a np matrix.
    After checking there is no obvious error in the problem (e.g two same digits
    in the same column), we solve the sudoku and return the solution.
    If no solution is found, we set the boolean self.found to False.
    Note that if the problem admits multiple solutions, we return the first one we have.
    """

    def __init__(self, matrix):
        self.grid = np.matrix(matrix)
        assert self.grid.shape == (9, 9)
        self.found = False
        self.is_possible = self.check_init()
        self.solution = self.grid
        self.duration = -1

    def propose_best(self):
        """
        Here the goal is to propose the optimal entry (x, y) to continue the recursion with.
        If the sudoku is filled with digits, we return None, None, []
        If the sudoku can be shortcuted, meaning we know that some entry can only contain
        a certain digit, we return the position and the corresponding digit.
        In all other cases, we propose the entry that has the lowest possible values,
        we return the position (x, y) and the possible values this entry can take.
        """

        # This is the most common/intuitive way to solve a sudoku by hand
        # You take each digit from 1 to 9, and examine if there is some place where this digit
        # must be
        # For this you remove from the possibilities every row, cols and sub-grid that
        # contain this digit
        for n in range(1, 10):
            shortcut = self.grid.copy()
            occurences = np.argwhere(shortcut == n)
            for occ in occurences:
                # Fill sub sudoku with 10 to mark the area where there can not be a "n"
                row_x, row_y = 3 * (occ[0] // 3), 3 * (occ[1] // 3)
                empty_spots = shortcut[row_x : row_x + 3, row_y : row_y + 3] == 0
                shortcut[row_x : row_x + 3, row_y : row_y + 3][empty_spots] = 10
                # Fill rows
                empty_spots = shortcut[occ[0], :] == 0
                shortcut[occ[0], :][empty_spots] = 10
                # Fill cols
                empty_spots = shortcut[:, occ[1]] == 0
                shortcut[:, occ[1]][empty_spots] = 10

            # Now check if we have a single empty spot qmong the sub-grids
            for i in range(3):
                for j in range(3):
                    zeros = np.argwhere(
                        shortcut[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] == 0
                    )
                    if len(zeros) == 1:
                        return 3 * i + zeros[0][0], 3 * j + zeros[0][1], [n]

        # If previous strategy didnt work, we must take some hypothesis
        # We look for the best candidate to consider, ie the entry that has the lowest amount
        # of possible values
        min_nb_poss = 9
        best_x, best_y = 0, 0
        filled = True
        for x in range(9):
            for y in range(9):
                if self.grid[x, y] == 0:
                    filled = False
                    nb_poss = np.count_nonzero(
                        [self.possible(x, y, n) for n in range(1, 10)]
                    )
                    if nb_poss < min_nb_poss:
                        min_nb_poss = nb_poss
                        best_x, best_y = x, y

                    if nb_poss == 0:
                        return -1, -1, []

        # no place anymore in sudoku --> this will cause the end of the recursion
        if filled:
            return None, None, []

        poss = [self.possible(best_x, best_y, n) for n in range(1, 10)]

        return (
            best_x,
            best_y,
            np.arange(1, 10, 1)[poss],
        )

    def check_init(self):
        """
        Here we check that there is no obvious error in the loaded problem
        """

        # Check rows
        for i in range(9):
            unique, count = np.unique(
                np.asarray(self.grid[i]).reshape(-1), return_counts=True
            )
            for j in np.argwhere(unique != 0):
                idx = j[0]
                nb_occ = count[idx]
                if nb_occ > 1:
                    print(f"ERROR: row {i} contains {nb_occ} `{unique[idx]}`")
                    return False

        # Check cols x
        for j in range(9):
            unique, count = np.unique(
                np.asarray(self.grid[:, j]).reshape(-1), return_counts=True
            )
            for i in np.argwhere(unique != 0):
                idx = i[0]
                nb_occ = count[idx]
                if nb_occ > 1:
                    print(f"ERROR: col {j} contains {nb_occ} `{unique[idx]}`")
                    return False

        # Check small grids
        for i in range(3):
            for j in range(3):

                unique, count = np.unique(
                    np.asarray(self.grid[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]).reshape(
                        -1
                    ),
                    return_counts=True,
                )
                for k in np.argwhere(unique != 0):
                    idx = k[0]
                    nb_occ = count[idx]
                    if nb_occ > 1:
                        print(
                            f"ERROR: one of the grid contains {nb_occ} `{unique[idx]}`"
                        )
                        return False

        return True

    def possible(self, x, y, n):
        """
        Determines if it s possible that entry grid[x, y] can take value n
        """

        # Check row x
        for i in range(9):
            if self.grid[i, y] == n:
                return False

        # Check col y
        for j in range(9):
            if self.grid[x, j] == n:
                return False

        # Check small grid around
        grid_start_x = (x // 3) * 3
        grid_start_y = (y // 3) * 3
        for i in range(grid_start_x, grid_start_x + 3):
            for j in range(grid_start_y, grid_start_y + 3):
                if self.grid[i, j] == n:
                    return False

        return True

    def solve_rec(self):
        """
        Here is the core of the recursion.
        Inspired from the video of Computerphile
        https://www.youtube.com/watch?v=G_UYXzGuqvM&ab_channel=Computerphile
        """

        # If a solution is found we must stop
        if self.found:
            return

        while True:
            x, y, possibilities = self.propose_best()
            if x is None:
                break

            if len(possibilities) == 1:
                self.grid[x, y] = possibilities[0]
                self.solve_rec()
                self.grid[x, y] = 0
                return

            if len(possibilities) == 0:
                return

            assert self.grid[x, y] == 0
            for n in possibilities:
                self.grid[x, y] = n
                self.solve_rec()
                self.grid[x, y] = 0

            return

        self.found = True
        self.solution = deepcopy(self.grid)

    def solve(self):
        """
        The acutal solve() that must be used as an API.
        Returns False if the sudoku was not solved.
        """

        if not self.is_possible:
            print("Dont solve sudoku since it s not possible")
            return False

        start = time.time()
        self.solve_rec()
        end = time.time()
        self.duration = end - start
        if self.found:
            print(f"Solved sudoku in {end - start:.3f} seconds")
            return True

        print("Could not solve sudoku")
        return False

    def display_problem(self):
        """Display problem contained in grid"""
        display(self.grid)

    def display_solution(self):
        """Display solution contained in self.solution"""
        display(self.solution)


def display(matrix):
    """Debuger/printer of sudoku"""
    print("+-------+-------+-------+")
    for ii in range(3):
        for i in range(3):
            print("|", end=" ")
            for jj in range(3):
                for j in range(3):
                    if matrix[3 * ii + i, 3 * jj + j] == 0:
                        print(" ", end=" ")
                    else:
                        print(matrix[3 * ii + i, 3 * jj + j], end=" ")
                print("|", end=" ")
            print("")
        print("+-------+-------+-------+")


if __name__ == "__main__":
    # Several examples of sudoku (different levels)
    # 0 means empty entry
    # You can observe how the solving duration changes with the difficulty
    easy = [
        [8, 0, 0, 0, 1, 0, 0, 0, 9],
        [0, 5, 0, 8, 0, 7, 0, 1, 0],
        [0, 0, 4, 0, 9, 0, 7, 0, 0],
        [0, 6, 0, 7, 0, 1, 0, 2, 0],
        [5, 0, 8, 0, 6, 0, 1, 0, 7],
        [0, 1, 0, 5, 0, 2, 0, 9, 0],
        [0, 0, 7, 0, 4, 0, 6, 0, 0],
        [0, 8, 0, 3, 0, 9, 0, 4, 0],
        [3, 0, 0, 0, 5, 0, 0, 0, 8],
    ]

    medium = [
        [4, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 9, 8],
        [3, 0, 0, 0, 8, 2, 4, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 8, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 6, 7, 0],
        [0, 5, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 9, 0, 7],
        [6, 4, 0, 3, 0, 0, 0, 0, 0],
    ]

    impossible = [
        [4, 4, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 9, 8],
        [3, 0, 0, 0, 8, 2, 4, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 8, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 6, 7, 0],
        [0, 5, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 9, 0, 7],
        [6, 4, 0, 3, 0, 0, 0, 0, 0],
    ]

    impossible_2 = [
        [8, 0, 0, 0, 9, 0, 0, 0, 9],
        [0, 5, 0, 8, 0, 7, 0, 1, 0],
        [0, 0, 4, 0, 9, 0, 7, 0, 0],
        [0, 6, 0, 7, 0, 1, 0, 2, 0],
        [5, 0, 8, 0, 6, 0, 1, 0, 7],
        [0, 1, 0, 5, 0, 2, 0, 9, 0],
        [0, 0, 7, 0, 4, 0, 6, 0, 0],
        [0, 8, 0, 3, 0, 9, 0, 4, 0],
        [3, 0, 0, 0, 5, 0, 0, 0, 8],
    ]

    hard = [
        [0, 6, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 9, 0],
        [0, 0, 0, 0, 1, 0, 8, 5, 0],
        [9, 1, 0, 3, 0, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 4, 1, 0],
        [0, 0, 5, 0, 0, 0, 0, 8, 9],
        [0, 4, 1, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 3],
    ]

    very_hard = [
        [0, 0, 0, 8, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 6, 0],
        [0, 0, 4, 0, 0, 0, 7, 0, 0],
        [5, 0, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 9, 6, 0, 0, 0],
        [8, 0, 0, 0, 0, 3, 0, 7, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 1, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 3, 8],
    ]

    sudoku = SudokuSolver(very_hard)

    sudoku.display_problem()
    if sudoku.is_possible:
        if sudoku.solve():
            sudoku.display_solution()
