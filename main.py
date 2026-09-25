"""Turn a landscape Rocket League recording into a finished portrait clip.

WHAT IT DOES
    Every frame goes through: 9:16 portrait crop -> boost gauge inset, but only
    on the frames that really show the gauge -> chat overlay, but only where the
    chat can actually be read out of the source -> optional title text.
    Audio is carried over from the source, trimmed the same way as the video.

    The three stages live in crop.py, boost.py and chat_overlay.py; this file
    runs them over a whole video and decides when each one applies.

NEEDS
    ffmpeg and ffprobe on PATH, plus opencv-python and numpy.
    artifacts/boost_overlay.png — the reference the boost check compares against.

BASIC USE
    python main.py videos/RL3.mp4
        Writes out/RL3_portrait.mp4. Chat is found automatically.

    python main.py videos/RL1.mp4 -o out/clip.mp4 --start 145 --duration 30
        Renders 30 seconds starting at 2:25. Use this while tuning — a full
        8-minute 4K source takes 10-15 minutes, 30 seconds takes about one.

    python main.py videos/RL1.mp4 --scan-only
        Prints the chat segments it found and stops, without rendering.

OPTIONS
    -o, --output PATH     where to write (default out/<name>_portrait.mp4)
    --start SECONDS       skip the first N seconds of the source
    --duration SECONDS    render only N seconds
    --title "a|b"         white title text over the first 3 seconds, "|" splits lines
    --chat off            skip chat entirely (also skips the chat scan, so it is faster)
    --chat-times "3-9,40-46"
                          use these ranges instead of searching for chat
    --no-boost            skip the boost inset
    --boost-threshold N   how sure the gauge match must be, see below
    --center-x 0-1        move the crop sideways, 0.5 is centred
    --crf N               x264 quality, lower is better, 18 is the default

HOW THE TWO AUTOMATIC DECISIONS WORK
    Boost — the gauge is missing during replays, goal cams and menus. Each frame's
    gauge corner is matched against artifacts/boost_overlay.png and the inset is
    drawn only above --boost-threshold (default 0.5). Measured on RL3, frames
    showing the gauge score 0.63-0.97 and everything else scores 0.37 or less,
    so 0.5 sits in the gap. Pass 0 to switch the check off and always draw.

    Chat — the chat box appears when someone types and fades out a few seconds
    later. The clip is walked in 1.5-second windows, every frame sampled, and
    chat_overlay.extract_chat is run on each window; whatever it returns is what
    gets shown, and where it finds nothing, nothing is drawn. A new overlay is
    extracted whenever the messages change, so the chat on screen keeps up with
    the conversation. It is drawn in the bottom black bar, left of the boost
    gauge, bottom-aligned so the newest message stays on the same line.

TUNING
    Everything else lives in the Settings dataclass just below: crop framing,
    where the boost inset and chat sit in the output frame, chat detection
    sensitivity, title size, encoder settings. Positions in output pixels are
    for a 1080x1920 frame; positions written as fractions are of the source.
    If you re-tune a stage in its notebook, copy the new numbers into Settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from chat_overlay import composite_chat, extract_chat, sample_chat
from crop import crop_portrait_better


# --------------------------------------------------------------------------
# Tunables. Fractions are of the source frame, pixels are in the output frame.
# --------------------------------------------------------------------------
@dataclass
class Settings:
    # Framing
    output_size: tuple[int, int] = (1080, 1920)
    center_x: float = 0.5
    wider_ratio: float = 1.0          # 1.0 = square image with black bars

    # Boost gauge inset
    boost_roi: tuple[float, float, float] = (0.915, 0.84, 0.12)  # cx, cy, radius
    boost_feather: int = 50           # source px, matches boost.crop_boost
    boost_position: tuple[int, int] = (656, 1526)
    boost_scale: float = 0.7
    # Every frame is matched against this reference gauge; below the threshold
    # the gauge is off screen (replay, menu, goal cam) and nothing is drawn.
    boost_reference: Path = Path(__file__).resolve().parent / "artifacts/boost_overlay.png"
    boost_threshold: float = 0.5      # 0 disables the check and always draws
    boost_probe: int = 96             # match at this size, plenty for the ring

    # Chat
    chat_roi: tuple[float, float, float, float] = (0.02, 0.033, 0.221, 0.22)
    chat_position: tuple[int, int] = (40, 1880)
    chat_anchor: str = "bottom"       # "bottom": y above is the text's bottom edge
    chat_width: int = 608             # 0.8x the full-size overlay
    chat_outline: int = 2
    chat_fade: float = 0.2            # seconds of ramp at each end
    chat_text_height: int = 36        # letter height in a 2160p source
    chat_samples: int = 0             # per overlay for --chat-times; 0 = every frame

    # Chat detection: the extractor itself decides, window by window
    chat_sample_fps: float = 0.0      # chat-box samples per second; 0 = every frame
    chat_window: float = 1.5          # seconds of samples behind one extraction
    chat_hop: float = 0.5             # how often a window is tested
    chat_threshold: int = 32          # top-hat contrast that counts as a stroke
    chat_extract_threshold: int = 45  # same, for extract_chat; higher = less scenery
    chat_min_pixels: int = 400        # glyph pixels below which we call it "no chat"
    chat_min_seconds: float = 0.8     # ignore shorter blips
    chat_change_iou: float = 0.45     # below this the messages changed -> re-extract

    # Title card
    title_seconds: float = 3.0
    title_scale: float = 2.2
    title_thickness: int = 5
    title_top: int = 150
    title_line_gap: int = 110

    # Encoding
    crf: int = 18
    preset: str = "medium"


# --------------------------------------------------------------------------
# ffmpeg plumbing
# --------------------------------------------------------------------------
def probe(path: Path) -> dict:
    """Return width, height, fps, duration and whether the file has audio."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    num, _, den = video["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    if fps <= 0:
        raise ValueError(f"Cannot read the frame rate of {path}")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": float(info["format"]["duration"]),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def read_frames(path: Path, width: int, height: int, start: float, duration: float | None):
    """Yield BGR source frames through an ffmpeg pipe."""
    command = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    command += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    frame_bytes = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=frame_bytes)
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        process.wait()


def open_writer(output: Path, size: tuple[int, int], fps: float, source: Path,
                start: float, duration: float | None, has_audio: bool, settings: Settings):
    """Start an ffmpeg encoder that takes raw frames on stdin and muxes source audio."""
    width, height = size
    command = [
        "ffmpeg", "-v", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "pipe:0",
    ]
    if has_audio:
        if start:
            command += ["-ss", f"{start:.3f}"]
        command += ["-i", str(source)]
        if duration is not None:
            command += ["-t", f"{duration:.3f}"]
        # No -shortest: both inputs are already trimmed, and it can cut the
        # encoder off a few frames before the last frame is written.
        command += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    command += [
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


# --------------------------------------------------------------------------
# Boost inset
# --------------------------------------------------------------------------
def blueness(bgr: np.ndarray) -> np.ndarray:
    """How blue a pixel is, the one thing the boost gauge always is."""
    return bgr[:, :, 0] - (bgr[:, :, 1] + bgr[:, :, 2]) / 2


class BoostStamp:
    """Feathered boost circle, drawn only on the frames that actually show it.

    The blend is boost.crop_boost + boost.composite_boost math, except the
    circular mask is built once and only the overlay's rectangle is touched
    instead of copying the whole 1080x1920 frame once per video frame.

    visible() matches each frame's gauge area against artifacts/boost_overlay.png
    and returns a -1..1 score. The comparison runs on "blueness", B - (R+G)/2,
    which is what the gauge's ring and digits are made of; raw pixels do not
    work because the gauge is translucent and its digits and orange ticks keep
    changing. Measured on RL3: every frame showing the gauge scores 0.63-0.97,
    every replay or menu frame scores 0.37 or less.
    """

    def __init__(self, src_width: int, src_height: int, settings: Settings):
        cx = int(src_width * settings.boost_roi[0])
        cy = int(src_height * settings.boost_roi[1])
        radius = int(min(src_width, src_height) * settings.boost_roi[2])
        x1, y1 = max(cx - radius, 0), max(cy - radius, 0)
        x2, y2 = min(cx + radius, src_width), min(cy + radius, src_height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Boost ROI falls outside the source frame")
        self.box = (x1, y1, x2, y2)

        yy, xx = np.ogrid[: y2 - y1, : x2 - x1]
        distance = np.hypot(xx - (cx - x1), yy - (cy - y1))
        if settings.boost_feather > 0:
            alpha = np.clip((radius - distance) / settings.boost_feather, 0.0, 1.0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
        else:
            alpha = (distance <= radius).astype(np.float32)

        scale = settings.boost_scale
        self.size = (max(1, round((x2 - x1) * scale)), max(1, round((y2 - y1) * scale)))
        self.interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        alpha = cv2.resize(alpha.astype(np.float32), self.size, interpolation=self.interpolation)
        self.alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
        self.position = settings.boost_position

        self.threshold = settings.boost_threshold
        self.probe = (settings.boost_probe, settings.boost_probe)
        self.reference = None
        if self.threshold > 0:
            reference = cv2.imread(str(settings.boost_reference), cv2.IMREAD_UNCHANGED)
            if reference is None:
                raise ValueError(f"Cannot read boost reference {settings.boost_reference}")
            small = cv2.resize(reference, self.probe, interpolation=cv2.INTER_AREA)
            self.weight = (small[:, :, 3].astype(np.float32) / 255 if small.shape[2] == 4
                           else np.ones(self.probe, np.float32))
            self.reference = self._centered(blueness(small[:, :, :3].astype(np.float32)))

    def _centered(self, values: np.ndarray) -> np.ndarray:
        """Subtract the alpha-weighted mean, so only the pattern is compared."""
        return values - (values * self.weight).sum() / self.weight.sum()

    def visible(self, source_frame: np.ndarray) -> float:
        """Weighted correlation between this frame's gauge area and the reference."""
        if self.reference is None:
            return 1.0
        x1, y1, x2, y2 = self.box
        patch = cv2.resize(source_frame[y1:y2, x1:x2], self.probe,
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        current = self._centered(blueness(patch))
        norms = np.sqrt((current * current * self.weight).sum()
                        * (self.reference * self.reference * self.weight).sum())
        if norms <= 0:
            return 0.0
        return float((current * self.reference * self.weight).sum() / norms)

    def apply(self, source_frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
        """Blend the gauge into canvas in place and return it."""
        x1, y1, x2, y2 = self.box
        patch = cv2.resize(source_frame[y1:y2, x1:x2], self.size,
                           interpolation=self.interpolation).astype(np.float32)
        left, top = self.position
        width, height = self.size
        bg_h, bg_w = canvas.shape[:2]
        if left >= bg_w or top >= bg_h or left + width <= 0 or top + height <= 0:
            return canvas
        # Clip against the canvas so an off-frame position degrades instead of crashing.
        sx1, sy1 = max(-left, 0), max(-top, 0)
        dx1, dy1 = max(left, 0), max(top, 0)
        dx2, dy2 = min(left + width, bg_w), min(top + height, bg_h)
        sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)

        alpha = self.alpha[sy1:sy2, sx1:sx2]
        region = canvas[dy1:dy2, dx1:dx2].astype(np.float32)
        blended = patch[sy1:sy2, sx1:sx2] * alpha + region * (1.0 - alpha)
        canvas[dy1:dy2, dx1:dx2] = np.rint(blended).clip(0, 255).astype(np.uint8)
        return canvas


# --------------------------------------------------------------------------
# Chat: find the stretches where it is on screen, then build one overlay each
# --------------------------------------------------------------------------
@dataclass
class ChatSegment:
    start: float                      # seconds, relative to the trimmed clip
    end: float
    overlay: np.ndarray | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


def source_chat_roi(width: int, height: int, settings: Settings) -> tuple[int, int, int, int]:
    left, top, right, bottom = settings.chat_roi
    return (int(width * left), int(height * top), int(width * right), int(height * bottom))


def chat_timeline(path: Path, roi: tuple[int, int, int, int], start: float,
                  duration: float | None, settings: Settings, text_height: float,
                  sample_fps: float) -> list[ChatSegment]:
    """Walk the whole clip and keep the chat only where it can be extracted.

    Decodes just the chat rectangle at sample_fps (every frame by default) and
    keeps a rolling window of samples. Every chat_hop seconds the window is tested: a white
    top-hat finds narrow bright strokes, and only strokes present across the
    whole window survive, so moving scenery drops out. When enough survive,
    extract_chat runs on those same samples and its overlay is what gets shown
    — if it comes back empty, that moment simply has no chat. The overlay is
    rebuilt whenever the surviving strokes change shape, which is what happens
    as messages arrive and expire.
    """
    x1, y1, x2, y2 = roi
    command = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    crop = f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}"
    if settings.chat_sample_fps > 0:
        crop += f",fps={settings.chat_sample_fps}"     # otherwise keep every frame
    command += ["-filter:v", crop, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    roi_w, roi_h = x2 - x1, y2 - y1
    frame_bytes = roi_w * roi_h * 3
    kernel_size = max(3, round(13 * text_height / 36) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,) * 2)

    step = 1.0 / sample_fps
    window = max(3, round(settings.chat_window * sample_fps))
    hop = max(1, round(settings.chat_hop * sample_fps))
    crops: deque[np.ndarray] = deque(maxlen=window)
    hats: deque[np.ndarray] = deque(maxlen=window)

    segments: list[ChatSegment] = []
    current: ChatSegment | None = None
    reference: np.ndarray | None = None

    def close(end: float) -> None:
        nonlocal current, reference
        if current is not None:
            current.end = end
            if current.duration >= settings.chat_min_seconds:
                segments.append(current)
        current, reference = None, None

    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=frame_bytes)
    try:
        index, now = 0, 0.0
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            crop = np.frombuffer(raw, np.uint8).reshape(roi_h, roi_w, 3)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crops.append(crop)
            hats.append(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
            index += 1
            if len(crops) < window or (index - window) % hop:
                continue

            now = (index - window / 2) * step        # centre of the window
            signature = np.min(np.stack(hats), axis=0) > settings.chat_threshold
            if int(signature.sum()) < settings.chat_min_pixels:
                close(now)
                continue

            union = int((signature | reference).sum()) if reference is not None else 0
            unchanged = union and int((signature & reference).sum()) / union >= settings.chat_change_iou
            if current is not None and unchanged:
                current.end = now + hop * step
                continue

            try:
                overlay, _ = extract_chat(np.stack(crops), text_height=max(4, round(text_height)),
                                          threshold=settings.chat_extract_threshold,
                                          edge_floor=settings.chat_extract_threshold // 2)
            except ValueError as error:
                print(f"  ! chat extraction failed at {now:.1f}s: {error}", file=sys.stderr)
                close(now)
                continue
            if int(np.count_nonzero(overlay[:, :, 3])) < settings.chat_min_pixels:
                close(now)                            # extractor found nothing usable
                continue

            close(now)
            current = ChatSegment(now, now + hop * step, scale_overlay(overlay, settings.chat_width))
            reference = signature
    finally:
        process.stdout.close()
        process.wait()

    close(now)
    return segments


def parse_ranges(text: str) -> list[ChatSegment]:
    """Parse "12.5-18,40-46" into segments."""
    segments = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        begin, _, finish = chunk.partition("-")
        segments.append(ChatSegment(float(begin), float(finish)))
    return segments


def scale_overlay(overlay: np.ndarray, width: int) -> np.ndarray:
    """Trim the blank rows, then resize BGRA in premultiplied space.

    Resizing premultiplied keeps transparent edges from turning dark. Trimming
    leaves a box that is exactly the text, so a bottom-anchored overlay keeps
    its newest message on the same line however many messages are on screen.
    """
    rows = np.flatnonzero(overlay[:, :, 3].any(axis=1))
    if rows.size:
        overlay = overlay[rows[0]:rows[-1] + 1]
    if width == overlay.shape[1]:
        return overlay
    alpha = overlay[:, :, 3].astype(np.float32) / 255
    premultiplied = overlay[:, :, :3].astype(np.float32) * alpha[:, :, None]
    height = max(1, round(overlay.shape[0] * width / overlay.shape[1]))
    interpolation = cv2.INTER_AREA if width < overlay.shape[1] else cv2.INTER_LINEAR
    premultiplied = cv2.resize(premultiplied, (width, height), interpolation=interpolation)
    alpha = cv2.resize(alpha, (width, height), interpolation=interpolation)
    safe = np.maximum(alpha, 1e-6)[:, :, None]
    color = np.where(alpha[:, :, None] > 0, premultiplied / safe, 0)
    return np.dstack((color.clip(0, 255).round().astype(np.uint8),
                      np.rint(alpha * 255).astype(np.uint8)))


def build_chat_overlays(path: Path, segments: list[ChatSegment], roi, start: float,
                        fps: float, settings: Settings, text_height: float) -> None:
    """Extract one chat overlay per segment, already scaled to output pixels."""
    for segment in segments:
        center = start + (segment.start + segment.end) / 2
        window = min(settings.chat_window, max(0.2, segment.duration * 0.8))
        count = settings.chat_samples or max(2, round(window * fps) + 1)
        try:
            samples, _ = sample_chat(path, round(center * fps), roi,
                                     window_seconds=window, sample_count=count)
            overlay, _ = extract_chat(samples, text_height=max(4, round(text_height)),
                                      threshold=settings.chat_extract_threshold,
                                      edge_floor=settings.chat_extract_threshold // 2)
        except (ValueError, RuntimeError) as error:
            print(f"  ! skipping chat at {segment.start:.1f}s: {error}", file=sys.stderr)
            continue
        if not overlay[:, :, 3].any():
            print(f"  ! no chat pixels survived at {segment.start:.1f}s", file=sys.stderr)
            continue
        segment.overlay = scale_overlay(overlay, settings.chat_width)


def apply_chat(canvas: np.ndarray, segment: ChatSegment, now: float, settings: Settings) -> np.ndarray:
    """Composite a segment's overlay, ramping the alpha at both ends."""
    overlay = segment.overlay
    if settings.chat_fade > 0:
        factor = min(1.0, (now - segment.start) / settings.chat_fade,
                     (segment.end - now) / settings.chat_fade)
        if factor <= 0:
            return canvas
        if factor < 1.0:
            overlay = overlay.copy()
            overlay[:, :, 3] = np.rint(overlay[:, :, 3] * factor).astype(np.uint8)
    left, y = settings.chat_position
    if settings.chat_anchor == "bottom":
        y = max(0, y - overlay.shape[0])
    return composite_chat(canvas, overlay, position=(left, y),
                          outline=settings.chat_outline)


# --------------------------------------------------------------------------
# Title card
# --------------------------------------------------------------------------
def draw_title(canvas: np.ndarray, lines: list[str], settings: Settings) -> None:
    """Draw centered white text with a black rim, in place."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for number, line in enumerate(lines):
        (text_w, _), _ = cv2.getTextSize(line, font, settings.title_scale, settings.title_thickness)
        origin = ((canvas.shape[1] - text_w) // 2,
                  settings.title_top + number * settings.title_line_gap)
        cv2.putText(canvas, line, origin, font, settings.title_scale, (0, 0, 0),
                    settings.title_thickness + 4, cv2.LINE_AA)
        cv2.putText(canvas, line, origin, font, settings.title_scale, (255, 255, 255),
                    settings.title_thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main pass
# --------------------------------------------------------------------------
def render(source: Path, output: Path, start: float, duration: float | None,
           segments: list[ChatSegment], title: list[str], settings: Settings, info: dict) -> int:
    width, height, fps = info["width"], info["height"], info["fps"]
    span = duration if duration is not None else max(0.0, info["duration"] - start)
    expected = round(span * fps)
    stamp = BoostStamp(width, height, settings) if settings.boost_scale > 0 else None
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(output, settings.output_size, fps, source, start, duration,
                         info["has_audio"], settings)

    index, gauge_frames = 0, 0
    begin = last_report = time.monotonic()
    try:
        for frame in read_frames(source, width, height, start, duration):
            now = index / fps
            canvas = crop_portrait_better(frame, output_size=settings.output_size,
                                          center_x=settings.center_x,
                                          wider_ratio=settings.wider_ratio)
            if stamp is not None and stamp.visible(frame) >= stamp.threshold:
                stamp.apply(frame, canvas)
                gauge_frames += 1
            for segment in segments:
                if segment.overlay is not None and segment.start <= now < segment.end:
                    canvas = apply_chat(canvas, segment, now, settings)
                    break
            if title and now < settings.title_seconds:
                draw_title(canvas, title, settings)

            writer.stdin.write(canvas.tobytes())
            index += 1

            if time.monotonic() - last_report >= 1.0:
                last_report = time.monotonic()
                rate = index / (last_report - begin)
                eta = (expected - index) / rate if rate > 0 else 0
                print(f"\r  {index}/{expected} frames  {rate:5.1f} fps  "
                      f"ETA {eta // 60:.0f}m{eta % 60:02.0f}s   ", end="", flush=True)
    except BrokenPipeError:
        print("\nffmpeg stopped early", file=sys.stderr)
    finally:
        if writer.stdin:
            writer.stdin.close()
        writer.wait()

    elapsed = max(time.monotonic() - begin, 1e-6)
    share = f", boost gauge on {100 * gauge_frames / index:.0f}% of them" if index else ""
    print(f"\r  {index} frames in {elapsed // 60:.0f}m{elapsed % 60:02.0f}s "
          f"({index / elapsed:.1f} fps){share}            ")
    if writer.returncode:
        raise RuntimeError(f"ffmpeg exited with {writer.returncode}")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="landscape recording to edit")
    parser.add_argument("-o", "--output", type=Path, help="default: out/<name>_portrait.mp4")
    parser.add_argument("--start", type=float, default=0.0, help="skip this many seconds")
    parser.add_argument("--duration", type=float, help="render only this many seconds")
    parser.add_argument("--chat", choices=("auto", "off"), default="auto",
                        help="auto-detect the chat box, or leave chat out")
    parser.add_argument("--chat-times", help='use these ranges instead, e.g. "12.5-18,40-46"')
    parser.add_argument("--scan-only", action="store_true",
                        help="print the detected chat segments and stop")
    parser.add_argument("--no-boost", action="store_true", help="leave the boost gauge out")
    parser.add_argument("--boost-threshold", type=float,
                        help="match score against artifacts/boost_overlay.png above which "
                             "the gauge counts as on screen (default 0.5, 0 always draws)")
    parser.add_argument("--title", help='title text for the first seconds, lines split on "|"')
    parser.add_argument("--center-x", type=float, help="horizontal crop center, 0-1")
    parser.add_argument("--crf", type=int, help="x264 quality, lower is better (default 18)")
    args = parser.parse_args(argv)

    if not args.source.exists():
        parser.error(f"{args.source} does not exist")

    settings = Settings()
    if args.center_x is not None:
        settings.center_x = args.center_x
    if args.crf is not None:
        settings.crf = args.crf
    if args.boost_threshold is not None:
        settings.boost_threshold = args.boost_threshold
    if args.no_boost:
        settings.boost_scale = 0.0

    info = probe(args.source)
    available = max(0.0, info["duration"] - args.start)
    duration = None if args.duration is None else min(args.duration, available)
    print(f"{args.source} — {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
          f"{info['duration']:.1f}s{'' if info['has_audio'] else ', no audio track'}")

    roi = source_chat_roi(info["width"], info["height"], settings)
    text_height = settings.chat_text_height * info["height"] / 2160

    segments: list[ChatSegment] = []
    if args.chat_times:
        segments = parse_ranges(args.chat_times)
        build_chat_overlays(args.source, segments, roi, args.start,
                            info["fps"], settings, text_height)
        segments = [s for s in segments if s.overlay is not None]
    elif args.chat == "auto":
        sample_fps = settings.chat_sample_fps or info["fps"]
        print(f"Looking for chat ({sample_fps:.0f} samples/s)...")
        segments = chat_timeline(args.source, roi, args.start, duration, settings,
                                 text_height, sample_fps)

    for segment in segments:
        print(f"  chat {segment.start:7.2f}s → {segment.end:7.2f}s  ({segment.duration:.2f}s)")
    if not segments:
        print("  no chat found")
    if args.scan_only:
        return 0

    output = args.output or Path("out") / f"{args.source.stem}_portrait.mp4"
    title = [part.strip() for part in args.title.split("|")] if args.title else []
    print(f"Rendering → {output}")
    render(args.source, output, args.start, duration, segments, title, settings, info)
    print(f"Done: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
