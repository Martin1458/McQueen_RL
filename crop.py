"""Portrait framing helpers for the Rocket League editing pipeline."""

import cv2
import numpy as np

def crop_portrait(frame, output_size=(1080, 1920), center_x=0.5):
    """Crop to the output aspect ratio, then resize; center_x is normalized."""
    out_width, out_height = output_size
    if out_width <= 0 or out_height <= 0 or not 0 <= center_x <= 1:
        raise ValueError("Invalid output size or horizontal center")
    height, width = frame.shape[:2]
    ratio = out_width / out_height
    if width / height >= ratio:
        crop_width = max(1, round(height * ratio))
        left = max(0, min(width - crop_width, round(center_x * width - crop_width / 2)))
        cropped = frame[:, left:left + crop_width]
    else:
        crop_height = max(1, round(width / ratio))
        top = (height - crop_height) // 2
        cropped = frame[top:top + crop_height]
    interpolation = cv2.INTER_AREA if cropped.shape[0] >= out_height else cv2.INTER_LINEAR
    return cv2.resize(cropped, output_size, interpolation=interpolation)

# crop_portrait_better is going to be a more advanced version of crop_portrait, 
# which crops a wider area and resizes it to fit on a 9:16 frame, 
# with black bars on top and bottom 


def crop_portrait_better(frame, output_size=(1080, 1920), center_x=0.5, wider_ratio=4/3):
    """Crop to a wider area, then resize to fit on a 9:16 frame with black bars."""
    out_width, out_height = output_size
    if out_width <= 0 or out_height <= 0 or not 0 <= center_x <= 1:
        raise ValueError("Invalid output size or horizontal center")
    height, width = frame.shape[:2]
    ratio = out_width / out_height
    # Crop a wider area (e.g., 4:3) instead of the final aspect ratio
    if wider_ratio <= 0:
        raise ValueError("Invalid wider ratio")
    if width / height >= wider_ratio:
        crop_width = max(1, round(height * wider_ratio))
        left = max(0, min(width - crop_width, round(center_x * width - crop_width / 2)))
        cropped = frame[:, left:left + crop_width]
    else:
        crop_height = max(1, round(width / wider_ratio))
        top = (height - crop_height) // 2
        cropped = frame[top:top + crop_height]
    # Resize to fit within the output size while maintaining aspect ratio
    interpolation = cv2.INTER_AREA if cropped.shape[0] >= out_height else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (out_width, int(out_width / wider_ratio)), interpolation=interpolation)
    # Create a black canvas and place the resized image in the center
    canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    y_offset = (out_height - resized.shape[0]) // 2
    canvas[y_offset:y_offset + resized.shape[0], :] = resized
    return canvas