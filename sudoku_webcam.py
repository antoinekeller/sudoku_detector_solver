#!/usr/bin/env python3

from attr import has
import cv2
import numpy as np
from solver import SudokuSolver
from sudoku_cv import Sudoku, clean_dilate_erode
from sudoku_locator import SudokuLocator
import time

# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_V4L2)

is_solved = False
sudoku = Sudoku()
is_solving = False


# def thread_solve_sudoku():
#    global sudoku
#    global is_solved
#    is_solved = sudoku.solve()
#    print("Sudoku is solved!")


text = "Sudoku solver"

center = [0, 0]

prev_time = None


while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    now = time.time()
    fps = int(1 / (now - prev_time)) if prev_time is not None else 0
    prev_time = now

    sudoku_loc = SudokuLocator(frame)

    has_sudoku = sudoku_loc.has_sudoku()
    has_sudoku_with_visibility = sudoku_loc.has_sudoku_with_visibility()

    print("##############")
    print("has sudoku : ", has_sudoku)
    print("is_solving :", is_solving)

    should_solve = False

    if has_sudoku_with_visibility:
        if np.linalg.norm(center - sudoku_loc.get_center()) < 2 and not is_solving:
            should_solve = True
        else:
            print("TOO SHAKY")
            should_solve = False
        center = sudoku_loc.get_center()

    print("should_solve : ", should_solve)

    if not has_sudoku:
        if is_solved:
            print("Reinitialize")
        is_solving = False
        is_solved = False

    if should_solve and not is_solving:
        text = "Detecting Sudoku"

        is_solved = False
        unwarped_img = sudoku_loc.unwarp()
        unwarped_img = clean_dilate_erode(unwarped_img)
        if not sudoku.infer_image(unwarped_img):
            text = "Bad detection"
            continue

        sudoku.solver = SudokuSolver(sudoku.sudoku)

        if not sudoku.solver.is_possible:
            text = "Bad detection"
            continue
        text = "Solving sudoku..."
        is_solving = True
        is_solved = sudoku.solve()

    if not has_sudoku:
        text = "No sudoku found"
        final_image = frame
    elif is_solved:
        sudoku_loc.render(sudoku)
        text = f"Solved in {sudoku.solver.duration:.3f}s"
        final_image = sudoku_loc.img
    else:
        final_image = sudoku_loc.get_image_with_corners()

    final_image = cv2.putText(
        final_image,
        text,
        org=(20, frame.shape[0] - 20),
        fontScale=1,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        color=(205, 204, 0),
        thickness=3,
    )
    final_image = cv2.putText(
        final_image,
        f"{fps}fps",
        org=(frame.shape[1] - 100, frame.shape[0] - 20),
        fontScale=1,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        color=(205, 204, 0),
        thickness=3,
    )
    cv2.imshow("Image", final_image)
    k = cv2.waitKey(1)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if k == 27:
        break
