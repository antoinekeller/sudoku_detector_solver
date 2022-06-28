"""
Define the class SudokuLocator
"""
import operator
import cv2
import numpy as np


class SudokuLocator:
    """
    Detect and locate a sudoku on an input image.
    Define if sudoku is sufficiently visible.
    Unwarp it to feed it later to our neural network.
    """

    def __init__(self, img, debug=False):
        self.img = img
        self.debug = debug
        self.img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Basic opencv functions to denoise and spot corners of the sudoku
        proc = cv2.GaussianBlur(self.img_grayscale.copy(), (9, 9), 0)
        proc = cv2.adaptiveThreshold(
            proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        proc = cv2.bitwise_not(proc, proc)
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
        proc = cv2.dilate(proc, kernel)

        # cv2.imshow("Processed image", proc)
        # cv2.waitKey(0)

        contours, _ = cv2.findContours(
            proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygon = contours[0]

        bottom_right, _ = max(
            enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
            key=operator.itemgetter(1),
        )
        top_left, _ = min(
            enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
            key=operator.itemgetter(1),
        )
        bottom_left, _ = min(
            enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
            key=operator.itemgetter(1),
        )
        top_right, _ = max(
            enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
            key=operator.itemgetter(1),
        )

        # 4 corners of the sudoku in the image coordinate
        self.corners = [
            polygon[top_left][0] + [5, 5],
            polygon[top_right][0] + [-5, 5],
            polygon[bottom_right][0] + [-5, -5],
            polygon[bottom_left][0] + [5, -5],
        ]

        src = np.array(self.corners, dtype="float32")
        self.side = max(
            [
                np.linalg.norm(self.corners[0] - self.corners[3]),
                np.linalg.norm(self.corners[1] - self.corners[2]),
                np.linalg.norm(self.corners[0] - self.corners[2]),
                np.linalg.norm(self.corners[1] - self.corners[3]),
            ]
        )

        dst = np.array(
            [
                [0, 0],
                [self.side - 1, 0],
                [self.side - 1, self.side - 1],
                [0, self.side - 1],
            ],
            dtype="float32",
        )

        # define warp matrix between original image and the sudoku submask
        self.warp_matrix = cv2.getPerspectiveTransform(src, dst)

    def has_four_corners_spread_out(self):
        """Check that the four corners belong to each quarter of image"""
        # top left
        if (
            self.corners[0][0] > self.img.shape[1] / 2
            or self.corners[0][1] > self.img.shape[0] / 2
        ):
            return False
        # top right
        if (
            self.corners[1][0] < self.img.shape[1] / 2
            or self.corners[1][1] > self.img.shape[0] / 2
        ):
            return False
        # bottom right
        if (
            self.corners[2][0] < self.img.shape[1] / 2
            or self.corners[2][1] < self.img.shape[0] / 2
        ):
            return False
        # bottom left
        if (
            self.corners[3][0] > self.img.shape[1] / 2
            or self.corners[3][1] < self.img.shape[0] / 2
        ):
            return False

        return True

    def has_sudoku_with_visibility(self):
        """
        Returns true if we consider that the sudoku is sufficiently visible,
        so that digits can be extracted safely
        """

        # Discard weird shapes
        if self.get_max_side() > 1.5 * self.get_min_side():
            return False

        # Discard sudoku too far from the camera
        if self.get_min_side() < 0.4 * self.img.shape[1]:
            return False

        if not self.has_sudoku():
            return False

        # Discard not-centered sudoku
        if not self.has_four_corners_spread_out():
            return False

        return True

    def has_sudoku(self):
        """
        The sudoku must be inside the camera
        Every corner at border of the camera is suspicious
        """

        for corner in self.corners:
            if corner[0] <= 5:
                return False
            if corner[0] >= self.img.shape[1] - 5:
                return False
            if corner[1] <= 5:
                return False
            if corner[1] >= self.img.shape[0] - 5:
                return False

        return True

    def get_center(self):
        """
        return center based on the middle of top left and bottom right corners
        """
        return (self.corners[0] + self.corners[2]) / 2

    def get_max_side(self):
        """
        returns the longest side of the sudoku
        """
        return np.max(
            [
                np.linalg.norm(self.corners[(i + 1) % 4] - self.corners[i])
                for i in range(4)
            ]
        )

    def get_min_side(self):
        """
        returns the shortest side of the sudoku
        """
        return np.min(
            [
                np.linalg.norm(self.corners[(i + 1) % 4] - self.corners[i])
                for i in range(4)
            ]
        )

    def get_image_with_corners(self):
        """
        Draw red dots at the corners location
        And return the new image
        """

        img_with_markers = self.img.copy()

        for corner in self.corners:
            img_with_markers = cv2.circle(
                img_with_markers,
                tuple(corner),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )

        return img_with_markers

    def unwarp(self):
        """Unwarp the image based on the warp matrix"""

        # Unwarp
        unwarped = cv2.warpPerspective(
            self.img_grayscale, self.warp_matrix, (int(self.side), int(self.side))
        )

        if self.debug:
            cv2.imshow("Warped image", unwarped)
            cv2.waitKey(0)

        return unwarped

    def render(self, sudoku):
        """
        Display solution on original image.
        Takes the sudoku problem/solution and display the solution on the empty entries.

        """

        solution_img = cv2.warpPerspective(
            self.img, self.warp_matrix, (int(self.side), int(self.side))
        )
        assert solution_img.shape[0] == solution_img.shape[1]
        dim = solution_img.shape[0] / 9
        solution_only = np.zeros(solution_img.shape, dtype="uint8")

        # Adapt font size to current width/height(=dim)
        fontscale = 0.025 * dim

        for i in range(9):
            for j in range(9):
                # Dont draw over existing digits, fill only empty entries
                if sudoku.digit_problem[j, i]:
                    continue

                pos_x = int(i * dim) + int(0.25 * dim)
                pos_y = int((j + 1) * dim) - int(0.25 * dim)
                solution_img = cv2.putText(
                    solution_img,
                    str(sudoku.solution[j, i]),
                    org=(pos_x, pos_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontscale,
                    color=(0, 255, 0),
                    thickness=2,
                )
                solution_only = cv2.putText(
                    solution_only,
                    str(sudoku.solution[j, i]),
                    org=(pos_x, pos_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontscale,
                    color=(0, 255, 0),
                    thickness=2,
                )

        # cv2.imshow("Result image", solution_img)
        # cv2.waitKey(0)

        # cv2.imshow("Result only", solution_only)
        # cv2.waitKey(0)

        # Rewarp the solution to the orginal image perspective
        _, m_inv = cv2.invert(self.warp_matrix)
        re_proj = cv2.warpPerspective(
            solution_only,
            m_inv,
            (self.img.shape[1], self.img.shape[0]),
        )
        # cv2.imshow("Unwarp image", re_proj)
        # cv2.waitKey(0)

        assert self.img.shape == re_proj.shape
        self.img = cv2.add(self.img, re_proj)

        # Add border lines
        for i in range(4):
            self.img = cv2.line(
                self.img,
                self.corners[i],
                self.corners[(i + 1) % 4],
                color=(0, 0, 255),
                thickness=2,
            )

        if self.debug:
            cv2.imshow("Final image", self.img)
            cv2.waitKey(0)
