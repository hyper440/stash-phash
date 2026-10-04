# stash-phash

A Python implementation of perceptual hash (pHash) generation that is fully compatible with [Stash](https://github.com/stashapp/stash).

## Features

- **Stash-compatible**: Generates identical phashes to Stash's built-in generator
- **Video support**: Extracts frames and creates sprite images for video hashing
- **Remote streams**: Supports HTTP, HTTPS, HLS (.m3u8), and other protocols via FFmpeg
- **Parallel extraction**: Uses concurrent frame extraction for faster processing of remote streams
- **CLI tool**: Easy command-line interface for generating hashes
- **Library**: Import and use in your own Python projects

## Installation

```bash
# Using uv (recommended)
uv pip install stash-phash

# Or install from source
git clone https://github.com/hyper440/stash-phash
cd stash-phash
uv pip install -e .
```

### Requirements

- Python 3.10+
- FFmpeg (for video processing) - must be in PATH or specified via `--ffmpeg`

## Usage

### Command Line

```bash
# Generate phash for a video file
stash-phash video.mp4

# Process multiple files
stash-phash video1.mp4 video2.mp4

# Remote HTTP/HTTPS video
stash-phash "https://example.com/video.mp4"

# HLS stream (.m3u8)
stash-phash "https://example.com/stream/playlist.m3u8"

# Output as integer (for database storage)
stash-phash --format int video.mp4

# Output as signed int64 (as stored in Stash's database)
stash-phash --format int-signed video.mp4

# Quiet mode (only output hashes, no filenames)
stash-phash -q video.mp4

# Verbose output (shows frame extraction progress)
stash-phash -v video.mp4

# Custom timeout for slow streams (default: 30s per frame)
stash-phash --timeout 60 "https://example.com/stream.m3u8"

# Specify custom FFmpeg path
stash-phash --ffmpeg /path/to/ffmpeg video.mp4
```

### Python Library

```python
from stash_phash import generate_video_phash, phash_to_string, hamming_distance

# Generate phash for a local video
phash = generate_video_phash("video.mp4")

# Generate phash for a remote stream
phash = generate_video_phash("https://example.com/stream.m3u8")

# Convert to hex string (Stash display format)
hex_string = phash_to_string(phash)
print(f"pHash: {hex_string}")  # e.g., "1a2b3c4d5e6f7890"

# Compare two phashes
phash1 = generate_video_phash("video1.mp4")
phash2 = generate_video_phash("video2.mp4")
distance = hamming_distance(phash1, phash2)

if distance < 10:
    print("Videos are likely duplicates!")
```

### Working with Images

```python
from PIL import Image
from stash_phash import perception_hash, hamming_distance

# Generate phash for an image
img = Image.open("image.jpg")
phash = perception_hash(img)

# Compare images
img1 = Image.open("image1.jpg")
img2 = Image.open("image2.jpg")
distance = hamming_distance(perception_hash(img1), perception_hash(img2))
```

## Algorithm

This implementation matches Stash's phash algorithm exactly:

### For Videos

1. Extract 25 frames (5x5 grid) from the video
2. Skip first/last 5% of duration to avoid intros/outros (the duration is rounded to 2 decimals, like Stash)
3. Each frame is scaled to 160 pixels wide (`scale=160:-2`, height keeps the aspect ratio rounded to a multiple of 2)
4. Frames are combined into a sprite image
5. Apply perception hash to the sprite

### Perception Hash (pHash)

1. Resize image to 64x64 using bilinear interpolation, ported from `nfnt/resize` (not PIL's `BILINEAR`, which is not bit-identical)
2. Convert to grayscale using luminosity formula: `0.299*R + 0.587*G + 0.114*B`
3. Apply 2D Discrete Cosine Transform (DCT-II)
4. Extract top-left 8x8 DCT coefficients (64 values)
5. Calculate median of these 64 values
6. Generate 64-bit hash: bit is 1 if coefficient > median, else 0

### Hamming Distance

The similarity between two phashes is measured using Hamming distance (number of differing bits):

- **0**: Identical images
- **1-10**: Very similar (likely duplicates)
- **10-20**: Somewhat similar
- **20+**: Different images

## Performance

### Local Files
- ~3-4 seconds for typical videos (fast seeking)

### Remote Streams (HLS, HTTP)
- Uses parallel extraction with 5 concurrent workers
- ~30-60 seconds for HLS streams (depending on network)
- ~2 minutes for very long videos (3+ hours)

### Resolution Independence
Higher resolution streams (1080p, 4K) produce identical hashes, while very low resolution
streams (144p) may differ slightly due to compression artifacts. Using a Hamming distance
threshold of 10-15 accounts for this variation.

## Compatibility

This implementation produces hashes that are:
- Identical to Stash's built-in phash generator (`phasher.exe`)
- Compatible with Stash's duplicate detection
- Storable in Stash's database format (signed int64)
- Displayable in Stash's hex format (16-character lowercase)

## Supported Protocols

Any protocol supported by FFmpeg, including:
- Local files (mp4, mkv, avi, webm, etc.)
- HTTP/HTTPS direct video URLs
- HLS streams (.m3u8 playlists)
- RTMP/RTMPS streams
- And more...

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/stash-phash
cd stash-phash

# Create virtual environment and install
uv venv
uv pip install -e .

# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run ruff format .
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Stash](https://github.com/stashapp/stash) - The original implementation
- [goimagehash](https://github.com/corona10/goimagehash) - Go library used by Stash
