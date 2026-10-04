"""Regression tests: output must be bit-identical to Stash (goimagehash / phasher)."""

import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from stash_phash.phash import perception_hash, resize_bilinear_nfnt
from stash_phash.video import extract_frame, stash_duration


def make_image(kind: str, seed: int, width: int, height: int) -> np.ndarray:
    """Deterministic test image (RandomState streams are stable across numpy versions)."""
    rs = np.random.RandomState(seed)
    if kind == "noise":
        return rs.randint(0, 256, size=(height, width, 3)).astype(np.uint8)
    small = rs.randint(0, 256, size=(12, 16, 3))
    ys = np.arange(height) * 12 // height
    xs = np.arange(width) * 16 // width
    img = small[ys][:, xs] + rs.randint(-12, 13, size=(height, width, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


# Expected values were produced by goimagehash.PerceptionHash (the code Stash uses) on an
# opaque NRGBA image built from exactly these pixels. Sizes mimic video sprites: 5 columns
# of 160px frames (width 800) and 5 rows of frames of varying height.
GO_VECTORS = [
    ("noise", 1, 800, 600, "8af598676d2416da"),
    ("noise", 2, 800, 585, "819dce318b7994ea"),
    ("noise", 3, 800, 1420, "c82335c49e4bb367"),
    ("noise", 4, 800, 450, "ace991a63e7466b0"),
    ("noise", 5, 800, 80, "865a5d5dba6192ca"),
    ("noise", 6, 640, 480, "c475b64f14199957"),
    ("blocks", 1, 800, 600, "889174dcd7f7c118"),
    ("blocks", 2, 800, 585, "d24bcd0f3c2073a7"),
    ("blocks", 3, 800, 1420, "ccc53587be5b4262"),
    ("blocks", 4, 800, 450, "a7ac62b4fa9f5012"),
    ("blocks", 5, 800, 80, "a85ed717dd30e160"),
    ("blocks", 6, 640, 480, "a256851d997db0b6"),
]


@pytest.mark.parametrize(("kind", "seed", "width", "height", "expected"), GO_VECTORS)
def test_perception_hash_matches_goimagehash(kind, seed, width, height, expected):
    img = Image.fromarray(make_image(kind, seed, width, height), "RGB")
    assert f"{perception_hash(img):016x}" == expected


def test_resize_returns_input_when_already_target_size():
    img = make_image("noise", 7, 64, 64)
    assert np.array_equal(resize_bilinear_nfnt(img), img)


def test_resize_output_shape_and_dtype():
    out = resize_bilinear_nfnt(make_image("blocks", 3, 800, 585))
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


class TestStashDuration:
    def test_rounds_to_two_decimals(self):
        assert stash_duration(53.053053) == 53.05
        assert stash_duration(41.775108) == 41.78

    def test_rounds_half_away_from_zero_like_go(self):
        # Go math.Round(12.5) == 13; Python round(12.5) == 12
        assert stash_duration(0.125) == 0.13


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_frame_height_is_multiple_of_two(tmp_path):
    """scale=160:-2 gives 160x118 for a 410x300 source; scale=160:-1 would give 160x117."""
    video = tmp_path / "t.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=410x300:rate=10",
         "-t", "1", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    assert extract_frame(video, 0.5).size == (160, 118)
