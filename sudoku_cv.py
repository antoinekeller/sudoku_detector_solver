#!/usr/bin/env python3

"""Define class Sudoku that extracts grid from a sudoku image
then solve it
"""

from typing import final
import cv2
import numpy as np
import operator
from tensorflow.keras.models import model_from_json
from solver import SudokuSolver
from sudoku_locator import SudokuLocator
from argparse import ArgumentParser
import time


def clean_dilate_erode(image):
    """Use basic opencv functions to transform incoming image"""

    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    image = cv2.bitwise_not(image)

    # Creating kernel
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)

    # Erode / Dilate
    image = cv2.dilate(image, kernel)
    image = cv2.erode(image, kernel)

    return image


class Sudoku:
    """Define sudoku class. Transform image into np array, then solve
    then display solution
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.sudoku = np.zeros((9, 9), dtype="uint8")

        # Load neural network model
        with open("ocr/model.json", "r", encoding="utf8") as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
            self.loaded_model = model_from_json(loaded_model_json)
            self.loaded_model.load_weights("ocr/model.h5")

    def infer_image(self, unwarped_img):
        """
        From an unwarped image, isolate each entry of the sudoku
        Then decide if the entry is blank or if it contains a digit
        Finally, feed the digit to the neural network.
        Returns True if everything went well.
        Returns False is one of the digits was badly recognized
        """

        entry_side = unwarped_img.shape[0] / 9
        margin = int(0.25 * entry_side)

        self.sudoku = np.zeros((9, 9), dtype="uint8")

        # numpy matrix containing grayscale image of each entry
        entries = np.array([[None] * 9] * 9, dtype=object)
        batch_imgs = None
        self.digit_problem = np.array([[False] * 9] * 9)
        for i in range(9):
            for j in range(9):
                width = int((i + 1) * entry_side) - int(i * entry_side)
                height = int((j + 1) * entry_side) - int(j * entry_side)
                entry = unwarped_img[
                    int(i * entry_side) : int((i + 1) * entry_side),
                    int(j * entry_side) : int((j + 1) * entry_side),
                ]

                # To determine if the entry is blank, perform floodfilling from every pixels
                # of the entry
                max_area = 0
                seed_point = (None, None)
                im_flood_fill = entry.copy()
                for x in range(margin, int(entry_side) - margin):
                    for y in range(margin, int(entry_side) - margin):
                        if entry.item(y, x) == 255:
                            area = cv2.floodFill(im_flood_fill, None, (x, y), 64)
                            if area[0] > max_area:
                                max_area = area[0]
                                seed_point = (x, y)

                # If the number of pixels building up the digit shape is too small, consider it as blank
                if max_area < 100:
                    entries[i, j] = None
                    continue

                cv2.floodFill(entry, None, seed_point, 64)

                # cv2.imshow("Digit", entry)
                # cv2.waitKey(0)

                # Resize the digit to 28x28 (neural network input shape)
                final_digit = np.zeros((width, height, 1), np.uint8)
                final_digit[entry == 64] = 255
                final_digit = cv2.resize(final_digit, dsize=(28, 28))

                # If there is no white pixel in the center of the image,
                # we can suspect that floodfilling was performed on the bounds of the entry
                # and that the entry is empty
                if len(np.argwhere(final_digit[9:19, 9:19] != 0)) == 0:
                    entries[i, j] = None
                    continue

                # cv2.imshow("Final", final_digit)
                # cv2.waitKey(0)

                # Center image
                start_x = np.argwhere(np.sum(final_digit, axis=0) != 0)
                start_y = np.argwhere(np.sum(final_digit, axis=1) != 0)
                if len(start_x) >= 2 and len(start_y) >= 2:
                    trans_x = (start_x[0][0] + start_x[-1][0]) // 2 - 14
                    trans_y = (start_y[0][0] + start_y[-1][0]) // 2 - 14

                    M = np.float32([[1, 0, -trans_x], [0, 1, -trans_y]])
                    final_digit = cv2.warpAffine(final_digit, M, final_digit.shape)

                entries[i, j] = final_digit

                # Transform uint8 numpy array to float array (values between 0 and 1)
                feed = np.asarray(final_digit) / 255

                # Append the final "image" to the batch images
                if batch_imgs is None:
                    batch_imgs = feed.reshape(1, 28, 28)
                else:
                    batch_imgs = np.concatenate(
                        (batch_imgs, feed.reshape(1, 28, 28)), axis=0
                    )

                self.digit_problem[i, j] = True

        # Measure time for neural network inference
        start = time.time()
        predictions = self.loaded_model.predict(
            batch_imgs.reshape(batch_imgs.shape[0], 28, 28, 1), verbose=0
        )
        end = time.time()
        print(
            f"Inference of {batch_imgs.shape[0]} digits done in {end-start:.3f} seconds"
        )

        batch_idx = 0
        for i in range(9):
            for j in range(9):
                if entries[i, j] is None:
                    continue

                prediction = np.argmax(predictions[batch_idx])
                if self.debug:
                    print(
                        f"Predicted digit: {prediction} with probability {predictions[batch_idx][prediction]:.2f}"
                    )

                # Declare problem if the output of the network is too low
                # If everything goes well, detection probability should be very close to 1
                if predictions[batch_idx][prediction] < 0.99:
                    print("Detection probability is too low, you should try again")
                    return False

                # There cant be a zero in a sudoku
                if prediction == 0:
                    print("Detected a ZERO...")
                    return False

                if self.debug:
                    cv2.imshow("Final", entries[i, j])
                    cv2.waitKey(0)

                self.sudoku[i, j] = prediction

                batch_idx += 1

        return True

    def solve(self):
        self.solver = SudokuSolver(self.sudoku)
        self.solver.display_problem()
        if self.solver.solve():
            self.solver.display_solution()
            self.solution = self.solver.solution
            return True
        return False


def main():
    # Load image
    img = cv2.imread(args.img)

    # cv2.imshow("Original image", img)
    # cv2.waitKey(0)

    # Localize sudoku in the image
    sudoku_loc = SudokuLocator(img, args.debug)

    # Unwarp/clean the image
    unwarped_img = sudoku_loc.unwarp()

    unwarped_img = clean_dilate_erode(unwarped_img)
    if args.debug:
        cv2.imshow("Cleaned image", unwarped_img)
        cv2.waitKey(0)

    sudoku = Sudoku()
    sudoku.infer_image(unwarped_img)

    sudoku.solve()

    sudoku_loc.render(sudoku)

    while True:
        cv2.imshow("Solved image", sudoku_loc.img)
        cv2.waitKey(1000)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Load an image, spot the sudoku in it, recognize the characters, then solve the sudoku",
    )
    parser.add_argument(
        "img",
        type=str,
        default="images/1.png",
        help="Path to sudoku image.",
    )
    parser.add_argument(
        "--debug",
        action="store_const",
        const=True,
        default=False,
        help="Display debug",
    )

    args = parser.parse_args()

    main()
