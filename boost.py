import cv2
import numpy as np


def composite_boost(background, overlay):
    if overlay.shape[-1] == 4:
        opacity = overlay[..., 3:4].astype(np.float32) / 255.0
        color = overlay[..., :3].astype(np.float32)
    else:
        opacity = np.ones((*overlay.shape[:2], 1), dtype=np.float32)
        color = overlay[..., :3].astype(np.float32)

    background = background.astype(np.float32)
    return np.rint(
        color * opacity + background * (1.0 - opacity)
    ).clip(0, 255).astype(np.uint8)

def crop_boost(orig_frame) -> tuple[np.ndarray, np.ndarray]:
    width, height = orig_frame.shape[1], orig_frame.shape[0]
    boost_roi = (int(width * 0.915), int(height * 0.84), int(min(width, height) * 0.12))
    cx, cy, r = boost_roi
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r, width), min(cy + r, height)

    boost_region = orig_frame[y1:y2, x1:x2]

    feather_px = 50
    yy, xx = np.ogrid[:boost_region.shape[0], :boost_region.shape[1]]
    distance = np.hypot(xx - (cx - x1), yy - (cy - y1))
    if feather_px > 0:
        alpha = np.clip((r - distance) / feather_px, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
    
    mask = np.rint(alpha * 255).astype(np.uint8)
    # Multiply by alpha: bitwise_and treats all nonzero mask values as opaque.
    boost_circular = np.rint(boost_region * alpha[..., None]).astype(np.uint8)

    return boost_circular, mask

def add_boost(orig_frame, cropped_frame, new_boost_pos = (726, 1566), new_boost_size = 1):
    boost_overlay, mask = crop_boost(orig_frame)

    # Composite the boost onto the cropped frame at the new position
    
    composite_frame = composite_boost(cropped_frame, boost_overlay)
    # Return the modified cropped frame
    return composite_frame

if __name__ == "__main__":
    # Example usage
    orig_frame = cv2.imread("middle_frame.png")
    cropped_frame = cv2.imread("portrait_better.png")
    new_boost_pos = (726, 1566)
    new_boost_size = 0.4

    result_frame = add_boost(orig_frame, cropped_frame, new_boost_pos, new_boost_size)
    cv2.imwrite("result_with_boost.png", result_frame)