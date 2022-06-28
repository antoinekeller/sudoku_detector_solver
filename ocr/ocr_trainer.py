#!/usr/bin/env python3
"""
Train a neural network on an auto-generated dataset to perform Optical Character Recognition.
First generate this datasets to have 28x28 grayscale images.
Then train a network and export it.
"""

from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt

import seaborn as sn
import pandas as pd


fonts = [
    "FranklinGothic.ttf",
    "rock.ttf",
    "verdana.ttf",
    "Cambria.ttf",
    "arial.ttf",
    "times.ttf",
    "Garamond.ttf",
    "futur.ttf",
    "calibri.ttf",
    "Helvetica 400.ttf",
]


def rotate_image(image, angle):
    """
    Use opencv to rotate image with a certain angle
    """

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def create_dataset():
    """
    Generate a dataset to be able to recognize printed digits.
    For this, we use classical fonts, and perform data augmentation by
    translation/scaling/rotating our digits from 0 to 9.
    Return the dataset as an array of 28x28 images (x) with the labels (y)
    """
    rotations = [-10, -5, 0, 5, 10]
    translations = [-2, -1, 0, 1, 2]

    font_sizes = [20, 22, 24, 26, 28]
    nb_images = (
        len(fonts) * len(font_sizes) * 10 * len(rotations) * len(translations) ** 2
    )

    print(
        f"Creating {nb_images} images with {len(fonts)} fonts, {len(rotations)} rotations, "
        f"{len(translations)}x{len(translations)} translations and {len(font_sizes)} "
        f"different font sizes"
    )
    x = np.zeros((nb_images, 28, 28))
    y = np.zeros(nb_images)
    img_idx = 0

    for font_name in fonts:
        unicode_text = "8"
        for font_size in font_sizes:
            font = ImageFont.truetype(f"fonts/{font_name}", font_size, encoding="unic")
            text_width, text_height = font.getsize(unicode_text)
            offset_xy = (14 - text_width / 2 + 1, 13 - text_height / 2)

            for i in range(10):
                canvas = Image.new("L", (28, 28), "black")

                # draw the text onto the text canvas
                draw = ImageDraw.Draw(canvas)
                draw.text(offset_xy, str(i), "white", font)

                open_cv_image = np.array(canvas)
                # cv2.imshow("OpenCV", open_cv_image)
                # cv2.waitKey(0)

                for rotation in rotations:

                    img_rotated = rotate_image(open_cv_image, rotation)
                    # cv2.imshow("rot", img_rotated)
                    # cv2.waitKey(0)

                    for translation_x in translations:
                        for translation_y in translations:
                            M = np.float32(
                                [[1, 0, translation_x], [0, 1, translation_y]]
                            )
                            img_translated = cv2.warpAffine(
                                img_rotated, M, dsize=img_rotated.shape
                            )
                            # cv2.imshow("rot+trans", img_translated)
                            # cv2.waitKey(0)

                            x[img_idx, :, :] = img_translated
                            y[img_idx] = i
                            img_idx += 1

    return x, y


def train(x, y):
    """
    Train a classical CNN network on the dataset
    Save the model to ocr/ (if --save)
    then display the confusion matrix
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42
    )

    print(
        f"Loaded training data with {x_train.shape[0]} images of dimensions "
        f"{x_train.shape[1]} x {x_train.shape[2]}"
    )
    print(
        f"Loaded test data with {x_test.shape[0]} images of dimensions "
        f"{x_test.shape[1]} x {x_test.shape[2]}"
    )

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # Create model

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()

    # Train

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.nb_epochs,
        batch_size=200,
    )
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f"CNN Error:{100 - scores[1] * 100:.2f}")
    print(model.predict(x_test[100:105]))

    # Test
    test_images = x_test[1:5]
    test_images = test_images.reshape(test_images.shape[0], 28, 28)
    print(f"Test images shape: {test_images.shape}")
    for i, test_image in enumerate(test_images, start=1):
        org_image = test_image
        test_image = test_image.reshape(1, 28, 28, 1)
        prediction = np.argmax(model.predict(test_image, verbose=0), axis=1)
        print(f"Predicted digit: {prediction[0]}")
        plt.subplot(220 + i)
        plt.axis("off")
        plt.title(f"Predicted digit: {prediction[0]}")
        plt.imshow(org_image.reshape((28, 28)), cmap=plt.get_cmap("gray"))
    plt.show()

    # Plot the confusion matrix
    test_predictions = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true=y_test, y_pred=test_predictions)

    df_cm = pd.DataFrame(
        cm, index=[i for i in "0123456789"], columns=[i for i in "0123456789"]
    )
    plt.figure(figsize=(20, 20))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    if args.save:
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w", encoding="utf-8") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a neural network on an auto-generated dataset to perform "
        "Optical Character Recognition",
    )
    parser.add_argument(
        "--save",
        action="store_const",
        const=True,
        default=False,
        help="Save model to disk",
    )
    parser.add_argument(
        "--nb-epochs",
        type=int,
        default=5,
        help="Number of epochs for training",
    )

    args = parser.parse_args()

    x, y = create_dataset()

    train(x, y)
