"""
    Run detector for each image in the directory.
"""

import os
import cv2
import imutils
from typing import TypedDict
from pathlib import Path

from core.detector import LFFDDetector

WALLPAPER_DIR = Path("~/Pictures/Wallpapers").expanduser()
VERT_WALLPAPER_DIR = Path("~/Pictures/WallpapersVertical").expanduser()


class Box(TypedDict):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float


def calculate_crop(image, boxes: list[Box]) -> tuple[Box, list[Box]]:
    height, width = image.shape[:2]
    target_width = int(height / 16 * 9)

    def clamp(xmin) -> tuple[int, int]:
        xmin = int(xmin)
        xmax = xmin + target_width

        # check if out of bounds and constrain it
        if xmin < 0:
            return 0, target_width
        elif xmax > width:
            return width - target_width, width
        else:
            return xmin, xmax

    if len(boxes) == 1:
        box = boxes[0]
        box_mid_x = (box["xmin"] + box["xmax"]) / 2
        target_x = box_mid_x - target_width / 2

        xmin, xmax = clamp(target_x)

        return (
            {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": 0,
                "ymax": height,
                "confidence": box["confidence"],
            },
            boxes,
        )

    else:
        # sort boxes by xmin
        boxes.sort(key=lambda box: box["xmin"])

        max_boxes = 0
        xmins = []

        for rect_left in range(width - target_width):
            rect_right = rect_left + target_width

            # check number of boxes in decimal within enclosed within larger rectangle
            num_boxes = 0
            for box in boxes:
                # no intersection, we overshot the final box
                if box["xmin"] > rect_right:
                    break

                # no intersection
                elif box["xmax"] < rect_left:
                    continue

                elif box["xmin"] >= rect_left and box["xmax"] <= rect_right:
                    num_boxes += 1
                    continue

                # partial intersection
                if box["xmin"] <= rect_right and box["xmax"] > rect_right:
                    num_boxes += (rect_right - box["xmin"]) / (
                        box["xmax"] - box["xmin"]
                    )
                    continue

            # update max boxes
            if num_boxes > 0:
                if num_boxes > max_boxes:
                    max_boxes = num_boxes
                    xmins = [rect_left]
                elif num_boxes == max_boxes:
                    xmins.append(rect_left)

        # get midpoint of start_x_arr
        start_x = xmins[len(xmins) // 2]
        xmin, xmax = clamp(start_x)

        return (
            {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": 0,
                "ymax": height,
                "confidence": boxes[0]["confidence"],
            },
            boxes,
        )


def write_cropped_image(image, boxes: list[Box], filename: str):
    rect, _ = calculate_crop(image, boxes)

    cropped = image[
        rect["ymin"] : rect["ymax"], rect["xmin"] : rect["xmax"]  # noqa: E203
    ]

    cv2.imwrite(str(VERT_WALLPAPER_DIR / filename), cropped)


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

    print("Start inferencing. Press `q` to cancel. Press  `-` to go back.")

    vertical_wallpapers = set(f.name for f in VERT_WALLPAPER_DIR.iterdir())

    for path in WALLPAPER_DIR.iterdir():
        # skip if already cropped
        if path.name in vertical_wallpapers:
            continue

        # use defaults
        image = cv2.imread(str(path))
        boxes = detector.detect(
            image,
            size=None,
            confidence_threshold=None,
            nms_threshold=None,
        )

        # skip if no boxes
        if not boxes:
            continue

        print(path, "x".join(image.shape[:2:-1]))

        # write to file
        write_cropped_image(image, boxes, path.name)

        # uncomment to show output
        # rect, detection_boxes = calculate_crop(image, boxes)
        # boxes_to_draw = [rect, *detection_boxes]

        # drawn_image = detector.draw(
        #     image,
        #     boxes_to_draw,
        #     # BGR
        #     color=(0, 0, 255),
        #     thickness=3,
        # )
        # drawn_image = imutils.resize(drawn_image, height=720)
        # cv2.imshow("Image", drawn_image)
        # key = cv2.waitKey(0) & 0xFF
        # # esc
        # if key == ord("q") or key == 27:
        #     break
        # # right arrow
        # elif key == ord("n") or key == 39:
        #     continue
