"""
Run detector for each image in the directory.
"""

import os
import cv2
from crop import WALLPAPER_DIR, VERT_WALLPAPER_DIR, write_cropped_image, calculate_crop

from core.detector import LFFDDetector

PREVIEW_IMAGES = False

if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    detector = LFFDDetector(
        {
            "lffd_config_path": "models/anime/lffd.json",
            "model_path": "models/anime/model.params",
            "symbol_path": "models/anime/symbol.json",
            "nms_threshold": 0.3,
            "confidence_threshold": 0.7,
            "size": 384,
        },
        use_gpu=False,
    )

    # skip images if already cropped
    vertical_wallpapers = set(f.name for f in VERT_WALLPAPER_DIR.iterdir())
    image_paths = [
        img for img in WALLPAPER_DIR.iterdir() if img.name not in vertical_wallpapers
    ]

    # uncomment to test specific images
    # image_paths = (
    #     WALLPAPER_DIR / f
    #     for f in [
    #         "wallhaven-l8dj1p.jpg",
    #     ]
    # )

    if PREVIEW_IMAGES:
        print("Start inferencing. Press `q` to cancel. Press  `p` to go back.")

    for path in image_paths:
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

        # display the images
        if PREVIEW_IMAGES:
            rect, detection_boxes = calculate_crop(image, boxes)
            boxes_to_draw = [rect, *detection_boxes]

            drawn_image = detector.draw(
                image,
                boxes_to_draw,
                # BGR
                color=(0, 255, 0),
                thickness=3,
            )

            w, h = image.shape[:2][::-1]
            resized_image = cv2.resize(drawn_image, (int(w / h * 720), 720))
            cv2.imshow("Image", resized_image)

            key = cv2.waitKey(0) & 0xFF
            # esc
            if key == ord("q") or key == 27:
                break
            # right arrow
            elif key == ord("n") or key == 39:
                continue
        # write to file
        else:
            write_cropped_image(image, boxes, path.name)
