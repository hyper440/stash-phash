"""
Video processing for perceptual hash generation.

This module handles extracting frames from videos and generating sprite images
for perceptual hashing, replicating the exact behavior of Stash's videophash
package.

Supports both local files and remote streams (HTTP, HTTPS, HLS, RTMP, etc.)
via FFmpeg's native protocol support.

The process:
1. Extract 25 frames (5x5 grid) from the video
2. Skip first 5% and last 5% of video duration (avoid intros/outros)
3. Each frame is resized to 160x160 pixels
4. Frames are combined into a single sprite image
5. Sprite is passed to the phash algorithm
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from stash_phash.phash import perception_hash

# Sprite configuration (matches Stash exactly)
SCREENSHOT_SIZE = 160
COLUMNS = 5
ROWS = 5
CHUNK_COUNT = COLUMNS * ROWS  # 25 frames

# Timeout for ffmpeg operations on streams (in seconds)
STREAM_TIMEOUT = 30

# Number of parallel frame extractions for remote streams
PARALLEL_EXTRACTIONS = 5


def is_url(path: str) -> bool:
    """
    Check if a path is a URL (remote stream).

    Args:
        path: Path or URL string

    Returns:
        True if the path is a URL, False otherwise
    """
    try:
        result = urlparse(str(path))
        return result.scheme in ("http", "https", "rtmp", "rtmps", "hls", "mms", "mmsh")
    except Exception:
        return False


def stash_duration(duration: float) -> float:
    """
    Round a duration the way Stash does before using it for frame timing.

    Stash stores ``FileDuration = math.Round(duration*100) / 100`` (pkg/ffmpeg/ffprobe.go)
    and derives all 25 seek times from that value. Go's math.Round rounds half away
    from zero, unlike Python's round().
    """
    return math.floor(duration * 100 + 0.5) / 100


def get_video_duration(
    video_source: str | Path,
    ffprobe_path: str | None = None,
    timeout: int | None = None,
) -> float:
    """
    Get the duration of a video file or stream in seconds.

    Args:
        video_source: Path to video file or URL to remote stream
        ffprobe_path: Optional path to ffprobe executable.
        timeout: Timeout in seconds for ffprobe

    Returns:
        Duration in seconds as a float
    """
    ffprobe = ffprobe_path or "ffprobe"
    source_str = str(video_source)
    is_remote = is_url(source_str)

    if timeout is None and is_remote:
        timeout = STREAM_TIMEOUT

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
    ]

    cmd.append(source_str)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    data = json.loads(result.stdout)

    # Get duration from format
    duration_str = data.get("format", {}).get("duration")

    if duration_str is None:
        raise ValueError(f"Could not determine duration for {video_source}")

    return float(duration_str)


def extract_frame(
    video_source: str | Path,
    timestamp: float,
    size: int = SCREENSHOT_SIZE,
    ffmpeg_path: str | None = None,
    timeout: int | None = None,
    output_path: str | None = None,
    slow_seek: bool = False,
) -> Image.Image:
    """
    Extract a single frame from a video at the specified timestamp.

    If the fast seek (``-ss`` before ``-i``) produces no frame, the extraction is retried
    once with an accurate seek (``-ss`` after ``-i``), as Stash does.

    Args:
        video_source: Path to video file or URL to remote stream
        timestamp: Time in seconds to extract frame from
        size: Width to resize the frame to
        ffmpeg_path: Optional path to ffmpeg executable
        timeout: Timeout in seconds
        output_path: Optional output path (uses temp file if None)
        slow_seek: Place -ss after -i (accurate seek). Normally only used as the fallback.

    Returns:
        PIL Image of the extracted frame
    """
    ffmpeg = ffmpeg_path or "ffmpeg"
    source_str = str(video_source)
    is_remote = is_url(source_str)

    if timeout is None:
        timeout = STREAM_TIMEOUT if is_remote else 60

    # Use provided path or temp file
    if output_path:
        tmp_path = output_path
        cleanup = False
    else:
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
            tmp_path = tmp.name
        cleanup = True

    try:
        seek = ["-ss", str(timestamp)]
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
        cmd += ([] if slow_seek else seek) + ["-i", source_str] + (seek if slow_seek else [])
        # scale=<w>:-2 (height rounded to a multiple of 2) is what Stash uses; -1 can give an
        # odd height, which changes the sprite size and therefore the hash.
        cmd += ["-frames:v", "1", "-vf", f"scale={size}:-2", "-update", "1", "-y", tmp_path]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)
            img = Image.open(tmp_path)
        except (subprocess.CalledProcessError, OSError):
            # Fast seek can fail or yield nothing (e.g. seeking near the end of the file).
            if slow_seek:
                raise
            return extract_frame(
                video_source, timestamp, size, ffmpeg_path, timeout, output_path, slow_seek=True
            )

        img_copy = img.copy()
        img.close()
        return img_copy

    finally:
        if cleanup:
            Path(tmp_path).unlink(missing_ok=True)


def extract_frames_parallel(
    video_source: str | Path,
    timestamps: list[float],
    size: int = SCREENSHOT_SIZE,
    ffmpeg_path: str | None = None,
    timeout: int | None = None,
    max_workers: int = PARALLEL_EXTRACTIONS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Image.Image]:
    """
    Extract multiple frames in parallel using ThreadPoolExecutor.

    This significantly speeds up extraction from remote streams by
    running multiple FFmpeg processes concurrently.

    Args:
        video_source: Path to video file or URL to remote stream
        timestamps: List of timestamps in seconds
        size: Width to resize frames to
        ffmpeg_path: Optional path to ffmpeg executable
        timeout: Timeout per extraction in seconds
        max_workers: Maximum parallel extractions
        progress_callback: Optional callback(completed, total)

    Returns:
        List of PIL Images in timestamp order
    """
    frames: dict[int, Image.Image] = {}
    completed = 0
    total = len(timestamps)

    def extract_single(idx: int, ts: float) -> tuple[int, Image.Image]:
        return idx, extract_frame(video_source, ts, size, ffmpeg_path, timeout)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single, i, ts): i for i, ts in enumerate(timestamps)}

        for future in as_completed(futures):
            idx, img = future.result()
            frames[idx] = img
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    # Return frames in order
    return [frames[i] for i in range(len(timestamps))]


def generate_sprite(
    video_source: str | Path,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    timeout: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Image.Image:
    """
    Generate a sprite image from video frames for perceptual hashing.

    This replicates Stash's sprite generation exactly:
    - Extracts 25 frames (5x5 grid)
    - Skips first and last 5% of video duration
    - Each frame is 160 pixels wide
    - Frames are arranged in a grid

    For remote streams, uses parallel extraction for speed.

    Args:
        video_source: Path to video file or URL to remote stream
        ffmpeg_path: Optional path to ffmpeg executable
        ffprobe_path: Optional path to ffprobe executable
        timeout: Timeout in seconds per frame
        progress_callback: Optional callback(frame_num, total_frames)

    Returns:
        PIL Image containing the sprite
    """
    source_str = str(video_source)
    is_remote = is_url(source_str)

    # Get video duration
    duration = stash_duration(get_video_duration(video_source, ffprobe_path, timeout))

    # Calculate frame timestamps (skip first/last 5%)
    offset = 0.05 * duration
    step_size = (0.9 * duration) / CHUNK_COUNT
    timestamps = [offset + (i * step_size) for i in range(CHUNK_COUNT)]

    # Extract frames
    if is_remote:
        # Use parallel extraction for remote streams
        frames = extract_frames_parallel(
            video_source,
            timestamps,
            SCREENSHOT_SIZE,
            ffmpeg_path,
            timeout,
            max_workers=PARALLEL_EXTRACTIONS,
            progress_callback=progress_callback,
        )
    else:
        # Sequential extraction for local files (fast seeking)
        frames = []
        for i, timestamp in enumerate(timestamps):
            if progress_callback:
                progress_callback(i + 1, CHUNK_COUNT)
            frame = extract_frame(video_source, timestamp, SCREENSHOT_SIZE, ffmpeg_path, timeout)
            frames.append(frame)

    return combine_images(frames)


def combine_images(images: list[Image.Image]) -> Image.Image:
    """
    Combine multiple images into a sprite grid.

    Args:
        images: List of PIL Images (should be 25 for video phash)

    Returns:
        PIL Image containing the combined sprite
    """
    if not images:
        raise ValueError("No images to combine")

    width = images[0].width
    height = images[0].height

    canvas_width = width * COLUMNS
    canvas_height = height * ROWS

    sprite = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    for index, img in enumerate(images):
        x = width * (index % COLUMNS)
        y = height * (index // COLUMNS)

        if img.mode != "RGBA":
            img = img.convert("RGBA")

        sprite.paste(img, (x, y))

    return sprite


def generate_video_phash(
    video_source: str | Path,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    timeout: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Generate a perceptual hash for a video file or remote stream.

    This is the main entry point for video phash generation. It:
    1. Generates a sprite image from video frames
    2. Computes the perceptual hash of the sprite

    The resulting hash is fully compatible with Stash's phash format.

    Supports:
    - Local video files (mp4, mkv, avi, webm, etc.)
    - Remote HTTP/HTTPS URLs
    - HLS streams (.m3u8)
    - RTMP streams
    - Any protocol supported by FFmpeg

    Args:
        video_source: Path to video file or URL to remote stream
        ffmpeg_path: Optional path to ffmpeg executable
        ffprobe_path: Optional path to ffprobe executable
        timeout: Timeout in seconds per frame (default: 30s for streams)
        progress_callback: Optional callback(frame_num, total_frames)

    Returns:
        64-bit perceptual hash as an unsigned integer

    Example:
        >>> # Local file
        >>> phash = generate_video_phash("video.mp4")
        >>> print(f"{phash:016x}")
        >>>
        >>> # HLS stream
        >>> phash = generate_video_phash("https://example.com/stream.m3u8")
        >>> print(f"{phash:016x}")
    """
    source_str = str(video_source)

    if not is_url(source_str):
        video_path = Path(video_source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

    sprite = generate_sprite(video_source, ffmpeg_path, ffprobe_path, timeout, progress_callback)
    return perception_hash(sprite)
