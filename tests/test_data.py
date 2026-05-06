from __future__ import annotations

from pathlib import Path

from src.data import discover_images, make_gallery_samples


def test_make_gallery_samples_marks_everything_as_gallery():
    pairs = [(Path("data/a/1.jpg"), "a"), (Path("data/b/2.jpg"), "b")]
    samples = make_gallery_samples(pairs)

    assert [s.split for s in samples] == ["gallery", "gallery"]
    assert [s.label for s in samples] == ["a", "b"]


def test_discover_images_uses_top_level_folder_as_label(tmp_path: Path):
    (tmp_path / "place_one").mkdir()
    (tmp_path / "place_two" / "nested").mkdir(parents=True)
    (tmp_path / "place_one" / "a.jpg").write_bytes(b"x")
    (tmp_path / "place_two" / "nested" / "b.jpeg").write_bytes(b"x")

    pairs = discover_images(tmp_path)

    assert [(path.name, label) for path, label in pairs] == [
        ("a.jpg", "place_one"),
        ("b.jpeg", "place_two"),
    ]


def test_discover_images_ignores_non_image_files(tmp_path: Path):
    (tmp_path / "place").mkdir()
    (tmp_path / "place" / "a.txt").write_text("x")
    (tmp_path / "place" / "b.jpg").write_bytes(b"x")

    pairs = discover_images(tmp_path)

    assert [(path.name, label) for path, label in pairs] == [("b.jpg", "place")]
