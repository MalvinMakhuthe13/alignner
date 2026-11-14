from pathlib import Path
import cv2
import numpy as np

from src.image_selector import IntelligentImageSelector


def test_selector_keeps_unique_images(tmp_path: Path):
    p1 = tmp_path / "one.jpg"
    p2 = tmp_path / "two.jpg"
    rng = np.random.default_rng(7)
    cv2.imwrite(str(p1), rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    cv2.imwrite(str(p2), rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    result = IntelligentImageSelector().analyze([str(p1), str(p2)])
    assert result.input_count == 2
    assert len(result.selected_paths) >= 1
    assert all(Path(p).exists() for p in result.selected_paths)


def test_unreadable_image_is_excluded(tmp_path: Path):
    p = tmp_path / "broken.jpg"
    p.write_text("not an image")
    result = IntelligentImageSelector().analyze([str(p)])
    assert result.assessments[0].decision == "exclude"
    assert result.excluded_paths == [str(p)]
