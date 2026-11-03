import unittest
from unittest.mock import patch

from services.hackmd_document import (
    HackMDImportError,
    _download_url,
    _image_map,
    normalize_hackmd_markdown,
)
from services.image_captioning import IMAGE_PATTERN


class FakeImage:
    size = (640, 480)


class HackMDDocumentTests(unittest.TestCase):
    def test_download_url_accepts_public_hackmd_view_link(self):
        self.assertEqual(
            _download_url("https://hackmd.io/@team/note-id?view"),
            "https://hackmd.io/@team/note-id/download",
        )
        with self.assertRaisesRegex(HackMDImportError, "只接受"):
            _download_url("https://example.com/note-id")

    def test_html_and_sized_markdown_images_are_discovered_in_source_order(self):
        markdown = normalize_hackmd_markdown(
            '# 架構\n<img alt="元件圖" src="https://hackmd.io/_uploads/a.png">\n'
            "## 流程\n![流程](https://hackmd.io/_uploads/b.jpg =500x)"
        )
        targets = [match.group("target") for match in IMAGE_PATTERN.finditer(markdown)]
        self.assertEqual(
            targets,
            [
                "https://hackmd.io/_uploads/a.png",
                "https://hackmd.io/_uploads/b.jpg",
            ],
        )

    def test_relative_and_repeated_images_are_downloaded_once(self):
        markdown = "![A](/_uploads/a.png)\n![A again](/_uploads/a.png)"
        with patch(
            "services.hackmd_document._load_image", return_value=FakeImage()
        ) as loader:
            images, downloads = _image_map(
                markdown, "https://hackmd.io/@team/note-id"
            )
        loader.assert_called_once_with("https://hackmd.io/_uploads/a.png")
        self.assertEqual(list(images), ["/_uploads/a.png"])
        self.assertEqual(downloads[0]["status"], "downloaded")

    def test_reference_style_image_is_normalized(self):
        markdown = normalize_hackmd_markdown(
            "![架構圖][arch]\n\n[arch]: https://hackmd.io/_uploads/arch.png"
        )
        self.assertIn(
            "![架構圖](https://hackmd.io/_uploads/arch.png)", markdown
        )


if __name__ == "__main__":
    unittest.main()
