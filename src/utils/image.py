"""Image processing utilities."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.core.types import Frame


class FrameSaver:
    """Save frames to a local directory.

    Example:
        saver = FrameSaver("./captured_frames")

        # Save a frame
        path = saver.save(frame)
        print(f"Saved to {path}")

        # Save with custom name
        path = saver.save(frame, name="alert_001")
    """

    def __init__(
        self,
        output_dir: str | Path = "./frames",
        format: str = "jpg",
        quality: int = 85,
        create_subdirs: bool = True,
    ):
        """Initialize frame saver.

        Args:
            output_dir: Directory to save frames
            format: Image format (jpg, png, webp)
            quality: JPEG/WebP quality (0-100)
            create_subdirs: Create date-based subdirectories (YYYY-MM-DD)
        """
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        self.quality = quality
        self.create_subdirs = create_subdirs
        self._frame_count = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        frame: Frame,
        name: str | None = None,
        metadata: dict | None = None,
    ) -> Path:
        """Save a frame to disk.

        Args:
            frame: Frame to save
            name: Optional custom filename (without extension)
            metadata: Optional metadata to save as JSON sidecar

        Returns:
            Path to saved file
        """
        # Determine output directory
        if self.create_subdirs:
            date_dir = self.output_dir / frame.timestamp.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            save_dir = date_dir
        else:
            save_dir = self.output_dir

        # Generate filename
        if name:
            filename = f"{name}.{self.format}"
        else:
            timestamp_str = frame.timestamp.strftime("%H%M%S_%f")[:-3]  # HHMMSSmmm
            self._frame_count += 1
            filename = f"frame_{timestamp_str}_{self._frame_count:04d}.{self.format}"

        filepath = save_dir / filename

        # Save image
        img = Image.fromarray(frame.data)

        save_kwargs = {}
        if self.format in ("jpg", "jpeg"):
            save_kwargs["quality"] = self.quality
            save_kwargs["optimize"] = True
        elif self.format == "webp":
            save_kwargs["quality"] = self.quality
        elif self.format == "png":
            save_kwargs["optimize"] = True

        img.save(filepath, **save_kwargs)

        # Save metadata sidecar if provided
        if metadata:
            import json
            meta_path = filepath.with_suffix(".json")
            full_metadata = {
                "timestamp": frame.timestamp.isoformat(),
                "source": frame.source,
                "shape": list(frame.data.shape),
                **metadata,
            }
            with open(meta_path, "w") as f:
                json.dump(full_metadata, f, indent=2)

        return filepath

    def save_batch(
        self,
        frames: list[Frame],
        prefix: str = "batch",
    ) -> list[Path]:
        """Save multiple frames.

        Args:
            frames: List of frames to save
            prefix: Filename prefix

        Returns:
            List of paths to saved files
        """
        paths = []
        for i, frame in enumerate(frames):
            path = self.save(frame, name=f"{prefix}_{i:04d}")
            paths.append(path)
        return paths

    @property
    def saved_count(self) -> int:
        """Number of frames saved."""
        return self._frame_count


def save_frame(
    frame: Frame,
    path: str | Path,
    format: str = "jpg",
    quality: int = 85,
) -> Path:
    """Save a single frame to a specific path.

    Args:
        frame: Frame to save
        path: Output path (with or without extension)
        format: Image format if not in path
        quality: JPEG/WebP quality

    Returns:
        Path to saved file
    """
    path = Path(path)

    # Add extension if not present
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    img = Image.fromarray(frame.data)

    save_kwargs = {}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality

    img.save(path, **save_kwargs)
    return path


def resize_frame(
    frame: Frame,
    max_size: int = 512,
    maintain_aspect: bool = True,
) -> Frame:
    """Resize a frame to fit within max_size.

    Args:
        frame: Frame to resize
        max_size: Maximum dimension (width or height)
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized Frame
    """
    img = Image.fromarray(frame.data)

    if maintain_aspect:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    else:
        img = img.resize((max_size, max_size), Image.Resampling.LANCZOS)

    return Frame(
        data=np.array(img),
        timestamp=frame.timestamp,
        source=frame.source,
    )


def frames_to_grid(
    frames: Sequence[Frame],
    columns: int = 2,
    max_size: int = 256,
    padding: int = 4,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Frame:
    """Combine multiple frames into a grid image.

    Useful for sending multiple frames as a single image to models
    that have limited image inputs.

    Args:
        frames: Frames to combine
        columns: Number of columns in grid
        max_size: Maximum size for each frame
        padding: Padding between frames
        background_color: Grid background color (RGB)

    Returns:
        Single Frame containing the grid
    """
    if not frames:
        raise ValueError("No frames to combine")

    # Resize all frames
    resized = [resize_frame(f, max_size) for f in frames]

    # Calculate grid dimensions
    rows = (len(resized) + columns - 1) // columns
    cell_width = max(f.shape[1] for f in resized)
    cell_height = max(f.shape[0] for f in resized)

    grid_width = columns * cell_width + (columns + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding

    # Create grid image
    grid = Image.new("RGB", (grid_width, grid_height), background_color)

    # Place frames in grid
    for i, frame in enumerate(resized):
        row = i // columns
        col = i % columns

        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)

        frame_img = Image.fromarray(frame.data)
        grid.paste(frame_img, (x, y))

    return Frame(
        data=np.array(grid),
        timestamp=frames[0].timestamp,
        source=f"grid:{len(frames)}_frames",
    )


def frame_to_bytes(
    frame: Frame,
    format: str = "JPEG",
    quality: int = 85,
) -> bytes:
    """Convert a frame to bytes.

    Args:
        frame: Frame to convert
        format: Image format (JPEG, PNG, etc.)
        quality: JPEG quality (0-100)

    Returns:
        Image bytes
    """
    img = Image.fromarray(frame.data)
    buffer = io.BytesIO()

    save_kwargs = {}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality

    img.save(buffer, format=format, **save_kwargs)
    return buffer.getvalue()


def bytes_to_frame(
    data: bytes,
    source: str = "bytes",
) -> Frame:
    """Convert bytes to a Frame.

    Args:
        data: Image bytes
        source: Source identifier

    Returns:
        Frame object
    """
    from datetime import datetime

    img = Image.open(io.BytesIO(data)).convert("RGB")

    return Frame(
        data=np.array(img),
        timestamp=datetime.now(),
        source=source,
    )


def draw_text_on_frame(
    frame: Frame,
    text: str,
    position: tuple[int, int] = (10, 10),
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int, int] | None = (0, 0, 0, 128),
) -> Frame:
    """Draw text on a frame.

    Args:
        frame: Frame to draw on
        text: Text to draw
        position: (x, y) position
        font_size: Font size
        color: Text color (RGB)
        background: Optional background color (RGBA)

    Returns:
        Frame with text overlay
    """
    from PIL import ImageDraw, ImageFont

    img = Image.fromarray(frame.data).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox(position, text, font=font)

    # Draw background if specified
    if background:
        padding = 4
        bg_bbox = (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        )
        draw.rectangle(bg_bbox, fill=background)

    # Draw text
    draw.text(position, text, font=font, fill=color)

    # Convert back to RGB
    result = img.convert("RGB")

    return Frame(
        data=np.array(result),
        timestamp=frame.timestamp,
        source=frame.source,
    )


def extract_frames_from_video(
    video_path: str,
    fps: float = 1.0,
    max_frames: int | None = None,
    max_size: int = 512,
) -> list[Frame]:
    """Extract frames from a video file.

    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        max_frames: Maximum number of frames
        max_size: Maximum frame dimension

    Returns:
        List of extracted Frames
    """
    import cv2
    from datetime import datetime, timedelta

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(video_fps / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_count = 0
    start_time = datetime.now()

    while cap.isOpened():
        ret, bgr_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to match desired fps
        if frame_count % frame_skip != 0:
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        frame = Frame(
            data=rgb_frame,
            timestamp=start_time + timedelta(seconds=frame_count / video_fps),
            source=f"video:{video_path}",
        )

        # Resize if needed
        frame = resize_frame(frame, max_size)
        frames.append(frame)

        # Check max frames
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames
