"""
    Run detector for each image in the directory.
"""

import os
import cv2
import imutils
from typing import TypedDict
from pathlib import Path

from core.detector import LFFDDetector


class Box(TypedDict):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float


def get_bounding_rect(image, boxes: list[Box]) -> Box:
    height, width = image.shape[:2]
    target_width = int(height / 16 * 9)

    # TODO: handle case with more than 1 box
    box = boxes[0]
    box_mid_x = (box["xmin"] + box["xmax"]) / 2
    target_x = box_mid_x - target_width / 2

    xmin = int(target_x)
    xmax = int(target_x + target_width)

    # check if out of bounds and constrain it
    if xmin < 0:
        xmin = 0
        xmax = target_width
    elif xmax > width:
        xmax = width
        xmin = width - target_width

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": 0,
        "ymax": height,
        "confidence": box["confidence"],
    }


WALLPAPER_DIR = Path("~/Pictures/Wallpapers").expanduser()
VERT_WALLPAPER_DIR = Path("~/Pictures/WallpapersVertical").expanduser()

if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    config = {
        "lffd_config_path": "models/anime/lffd.json",
        "model_path": "models/anime/model.params",
        "symbol_path": "models/anime/symbol.json",
        "nms_threshold": 0.3,
        "confidence_threshold": 0.7,
        "size": 384,
    }

    detector = LFFDDetector(config, use_gpu=False)

    # print("Start inferencing. Press `q` to cancel. Press  `-` to go back.")

    for path in WALLPAPER_DIR.iterdir():
        image = cv2.imread(str(path))

        # use defaults
        boxes = detector.detect(
            image,
            size=None,
            confidence_threshold=None,
            nms_threshold=None,
        )

        # single waifu
        if len(boxes) == 1:
            rect = get_bounding_rect(image, boxes)

            cropped_image = image[
                rect["ymin"] : rect["ymax"], rect["xmin"] : rect["xmax"]  # noqa: E203
            ]

            cv2.imwrite(str(VERT_WALLPAPER_DIR / path.name), cropped_image)

        # if len(boxes) > 1:
        #     # final_box = get_bounding_rect(image, boxes)

        #     print(path)
        #     print(image.shape[:2:-1])
        #     print(boxes)

        #     drawn_image = detector.draw(image, boxes, color=(0, 0, 255), thickness=3)
        #     drawn_image = imutils.resize(drawn_image, height=720)
        #     cv2.imshow("Image", drawn_image)
        #     key = cv2.waitKey(0) & 0xFF
        #     if key == ord("q") or key == 27:
        #         break
        #     else:
        #         continue

    # paths = WALLPAPER_DIR.iterdir()
    # idx = 0
    # while True:
    #     path = paths[idx]
    #     image = cv2.imread(path)
    #     start = time.time()
    #     # use defaults
    #     boxes = detector.detect(
    #         image,
    #         size=None,
    #         confidence_threshold=None,
    #         nms_threshold=None,
    #     )
    #     elapsed = time.time() - start
    #     drawn_image = detector.draw(image, boxes, color=(0, 0, 255), thickness=3)
    #     drawn_image = imutils.resize(drawn_image, height=720)
    #     cv2.imshow("Image", drawn_image)
    #     print(
    #         f"{time.ctime()}: [{idx:>7}] Inference time: {elapsed:.4f} seconds. Path: {path}"
    #     )
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == ord("q"):
    #         break
    #     elif key == ord("p"):
    #         idx = max(idx - 1, 0)
    #     else:
    #         idx += 1
    #         if idx >= len(paths):
    #             break
