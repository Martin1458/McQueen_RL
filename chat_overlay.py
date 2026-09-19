"""Extract existing chat glyphs as BGRA pixels without recognizing the text.

Temporal extraction assumes the chat stays in the same position and contains
the same messages throughout the sample window. Use short windows near updates.
"""

import cv2
import numpy as np


def sample_chat(video_path, frame_index, roi, window_seconds=2.0, sample_count=41):
    """Read a centered window sequentially; return BGR crops and frame indices.

    ROI coordinates are (left, top, right, bottom) in source-video pixels.
    Only the crops are retained in memory, even for 4K sources.
    """
    if window_seconds < 0 or sample_count < 1:
        raise ValueError("Use a nonnegative window and positive sample count")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("ROI must lie inside the source frame")
        if fps <= 0 or not 0 <= frame_index < count:
            raise ValueError("Invalid frame index or video timing")
        radius = round(window_seconds * fps / 2)
        start, stop = max(0, frame_index - radius), min(count - 1, frame_index + radius)
        wanted = set(np.linspace(start, stop, min(sample_count, stop - start + 1)).round().astype(int))
        wanted.add(frame_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        crops, indices = [], []
        for index in range(start, stop + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index}")
            if index in wanted:
                crops.append(frame[y1:y2, x1:x2].copy())
                indices.append(index)
        return np.stack(crops), indices
    finally:
        cap.release()


def extract_chat(samples, text_height=36, percentile=35, threshold=32,
                 edge_floor=16, opaque_contrast=71):
    """Return (BGRA overlay, debug images) from a stable chat window.

    A white top-hat measures narrow bright strokes against their local
    background. Taking its temporal percentile removes strokes that only
    occur briefly as the scene moves. Small connected-component filtering
    removes remaining specks. Colors come from the temporal median image.

    Defaults target ~36-pixel-high letters in the notebook's 4K source.
    This is an approximate soft matte, not recovery of the game's true alpha.
    A still background, scene cut, fade, or chat update can produce artifacts.
    """
    samples = np.asarray(samples)
    if samples.ndim != 4 or samples.shape[-1] != 3 or not samples.shape[0]:
        raise ValueError("Expected samples with shape (frames, height, width, 3)")
    if samples.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR samples")
    if text_height <= 0 or not 0 <= percentile <= 100:
        raise ValueError("Invalid text height or percentile")
    if not 0 <= edge_floor < threshold < opaque_contrast <= 255:
        raise ValueError("Require 0 <= edge_floor < threshold < opaque_contrast <= 255")
    scale = text_height / 36
    diameter = max(3, round(13 * scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    contrast = np.stack([
        cv2.morphologyEx(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, kernel)
        for frame in samples
    ])
    persistent = np.percentile(contrast, percentile, axis=0)
    seeds = (persistent > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds)
    keep = np.zeros(count, np.uint8)
    for label, (_, _, width, height, area) in enumerate(stats[1:], 1):
        if (area >= max(2, 8 * scale**2) and max(2, 5 * scale) <= height <= 45 * scale
                and width <= 50 * scale):
            keep[label] = 1
    # Expand seed support by one source pixel to retain antialiased edges.
    edge_radius = max(1, round(scale))
    support = cv2.dilate(keep[labels], np.ones((2 * edge_radius + 1,) * 2, np.uint8))
    alpha = np.clip((persistent - edge_floor) / (opaque_contrast - edge_floor), 0, 1)
    alpha = np.rint(alpha * support * 255).astype(np.uint8)
    median = np.median(samples, axis=0).astype(np.uint8)
    overlay = np.dstack((median, alpha))
    overlay[alpha == 0, :3] = 0
    return overlay, {"median": median, "contrast": persistent, "alpha": alpha}


def composite_chat(frame, overlay, position=(0, 0), width=None, outline=0):
    """Alpha-composite BGRA chat onto BGR video; optionally resize and outline.

    Resize in premultiplied-alpha space so transparent edges do not turn dark.
    Position and outline radius are measured in output-frame pixels.
    """
    x, y = position
    if x < 0 or y < 0 or outline < 0 or (width is not None and width < 1):
        raise ValueError("Invalid overlay position, width, or outline")
    alpha = overlay[:, :, 3].astype(np.float32) / 255
    premultiplied = overlay[:, :, :3].astype(np.float32) * alpha[:, :, None]
    if width is not None and width != overlay.shape[1]:
        height = max(1, round(overlay.shape[0] * width / overlay.shape[1]))
        interpolation = cv2.INTER_AREA if width < overlay.shape[1] else cv2.INTER_LINEAR
        premultiplied = cv2.resize(premultiplied, (width, height), interpolation=interpolation)
        alpha = cv2.resize(alpha, (width, height), interpolation=interpolation)
    height, width = alpha.shape
    if x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("Overlay must fit inside the output frame")
    result = frame.copy()
    region = result[y:y + height, x:x + width].astype(np.float32)
    if outline:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1,) * 2)
        shadow = cv2.dilate(alpha, kernel)
        region *= 1 - shadow[:, :, None]
    result[y:y + height, x:x + width] = np.clip(
        premultiplied + region * (1 - alpha[:, :, None]), 0, 255
    ).round().astype(np.uint8)
    return result
