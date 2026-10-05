from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePath

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}


def _probe_duration(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for video frame extraction.")

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffprobe timed out while inspecting the video.") from exc
    except OSError as exc:
        raise RuntimeError("Could not start ffprobe.") from exc

    if result.returncode != 0:
        raise ValueError("The uploaded video could not be opened.")
    try:
        duration = float(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("The uploaded video has no readable duration.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The uploaded video has no readable duration.")
    return duration


def sample_video_frames(
    video_path: Path,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
    max_dimension: int,
    jpeg_quality: int,
    timeout_seconds: float,
    duration_seconds: float | None = None,
) -> tuple[float, list[tuple[float, bytes]]]:
    """Sample evenly spaced frames across the clip, bounded by max_frames."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for video frame extraction.")
    if not video_path.is_file():
        raise ValueError("The uploaded video could not be opened.")
    if fps <= 0 or min_frames < 1 or max_frames < 1 or max_dimension < 1:
        raise ValueError("Invalid video sampling settings.")

    duration = duration_seconds if duration_seconds is not None else _probe_duration(video_path)
    frame_count = min(max_frames, max(min_frames, math.ceil(duration * fps)))
    frames: list[tuple[float, bytes]] = []

    with tempfile.TemporaryDirectory(prefix="ollama_video_frames_") as frame_directory:
        for index in range(frame_count):
            timestamp = duration * (index + 0.5) / frame_count
            frame_path = Path(frame_directory) / f"frame_{index:02d}.jpg"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-vf",
                f"scale={max_dimension}:{max_dimension}:force_original_aspect_ratio=decrease",
                "-frames:v",
                "1",
                "-q:v",
                str(jpeg_quality),
                str(frame_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"ffmpeg timed out extracting frame {index + 1}.") from exc
            except OSError as exc:
                raise RuntimeError("Could not start ffmpeg.") from exc

            if result.returncode == 0 and frame_path.is_file():
                frames.append((timestamp, frame_path.read_bytes()))

    if not frames:
        raise ValueError("No frames could be extracted from the uploaded video.")
    return duration, frames


def extract_video_frames(
    data: bytes,
    filename: str,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
    max_dimension: int,
    jpeg_quality: int,
    timeout_seconds: float,
    max_duration_seconds: float,
) -> tuple[float, list[tuple[float, bytes]]]:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {suffix or '(no extension)'}.")
    if not data:
        raise ValueError("The uploaded video is empty.")

    with tempfile.TemporaryDirectory(prefix="ollama_video_input_") as directory:
        video_path = Path(directory) / f"input{suffix}"
        video_path.write_bytes(data)
        duration = _probe_duration(video_path)
        if duration > max_duration_seconds:
            raise ValueError(f"Video exceeds the {max_duration_seconds:g}-second limit.")
        return sample_video_frames(
            video_path,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
            timeout_seconds=timeout_seconds,
            duration_seconds=duration,
        )
