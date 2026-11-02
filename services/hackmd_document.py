"""Download a public HackMD note and enrich every referenced image in place."""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from services.image_captioning import (
    IMAGE_PATTERN,
    ExternalMultimodalCaptioner,
    ImageCaptionSettings,
    enrich_markdown_with_marker_images,
)
from services.project_security import validate_project_document_relevance


MAX_MARKDOWN_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGES = 40
HACKMD_HOSTS = {"hackmd.io", "www.hackmd.io"}
HTML_IMAGE_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
HTML_ATTRIBUTE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s>]+)",
    re.IGNORECASE,
)
REFERENCE_DEFINITION_PATTERN = re.compile(
    r"(?m)^\s*\[(?P<label>[^\]]+)\]:\s*"
    r"(?P<target><[^>]+>|\S+)(?:\s+[\"'][^\"']*[\"'])?\s*$"
)
REFERENCE_IMAGE_PATTERN = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\[(?P<label>[^\]]*)\]"
)


class HackMDImportError(RuntimeError):
    pass


def _hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").rstrip(".").lower()


def _validate_hackmd_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or _hostname(url) not in HACKMD_HOSTS:
        raise HackMDImportError("只接受 https://hackmd.io 的公開筆記連結。")
    if parsed.username or parsed.password or not parsed.path.strip("/"):
        raise HackMDImportError("HackMD 連結格式不正確。")


def _validate_public_image_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise HackMDImportError(f"圖片不是可下載的 HTTP(S) 網址：{url[:160]}")
    if parsed.username or parsed.password:
        raise HackMDImportError("圖片網址不得包含帳號或密碼。")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
            )
        }
    except socket.gaierror as exc:
        raise HackMDImportError(f"無法解析圖片主機：{parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HackMDImportError("圖片網址指向內部或保留網路位址，已拒絕下載。")


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], None]) -> None:
        super().__init__()
        self.validator = validator

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.validator(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_response(response, limit: int) -> bytes:
    length = response.headers.get("Content-Length")
    if length and int(length) > limit:
        raise HackMDImportError(f"遠端內容超過大小限制（{limit // 1024 // 1024} MiB）。")
    payload = response.read(limit + 1)
    if len(payload) > limit:
        raise HackMDImportError(f"遠端內容超過大小限制（{limit // 1024 // 1024} MiB）。")
    return payload


def _fetch(
    url: str,
    *,
    limit: int,
    validator: Callable[[str], None],
) -> tuple[bytes, str, str]:
    validator(url)
    opener = build_opener(_ValidatedRedirectHandler(validator))
    request = Request(
        url,
        headers={
            "User-Agent": "SE-Mentor-HackMD-Importer/1.0",
            "Accept": "text/markdown,text/plain,image/*,application/octet-stream;q=0.8",
        },
    )
    try:
        with opener.open(request, timeout=30) as response:
            validator(response.geturl())
            return (
                _read_response(response, limit),
                response.headers.get_content_type(),
                response.geturl(),
            )
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise HackMDImportError(f"下載失敗：{url[:160]}（{exc}）") from exc


def _download_url(note_url: str) -> str:
    _validate_hackmd_url(note_url)
    parsed = urlsplit(note_url)
    path = re.sub(r"/(?:edit|view|publish|slide|download)/?$", "", parsed.path)
    path = path.rstrip("/") + "/download"
    return urlunsplit(("https", "hackmd.io", path, "", ""))


def download_hackmd_markdown(note_url: str) -> tuple[str, str]:
    """Return raw Markdown and the canonical response URL for a public note."""
    payload, content_type, response_url = _fetch(
        _download_url(note_url),
        limit=MAX_MARKDOWN_BYTES,
        validator=_validate_hackmd_url,
    )
    markdown = payload.decode("utf-8-sig")
    prefix = markdown.lstrip()[:200].lower()
    if "html" in content_type or prefix.startswith(("<!doctype html", "<html")):
        raise HackMDImportError(
            "HackMD 沒有回傳 Markdown；請確認筆記已設為所有人可閱讀。"
        )
    if not markdown.strip():
        raise HackMDImportError("HackMD 筆記內容為空。")
    return markdown, response_url


def _html_images_to_markdown(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        attrs = {}
        for attr in HTML_ATTRIBUTE_PATTERN.finditer(match.group(0)):
            value = attr.group("value").strip("\"'")
            attrs[attr.group("name").lower()] = html.unescape(value)
        source = attrs.get("src")
        if not source:
            return match.group(0)
        alt = attrs.get("alt", "圖片").replace("]", "")
        return f"![{alt}]({source})"

    return HTML_IMAGE_PATTERN.sub(replace, markdown)


def _reference_images_to_inline(markdown: str) -> str:
    definitions = {
        match.group("label").strip().casefold(): match.group("target").strip("<>")
        for match in REFERENCE_DEFINITION_PATTERN.finditer(markdown)
    }

    def replace(match: re.Match[str]) -> str:
        label = (match.group("label") or match.group("alt")).strip().casefold()
        target = definitions.get(label)
        if not target:
            return match.group(0)
        return f"![{match.group('alt')}]({target})"

    return REFERENCE_IMAGE_PATTERN.sub(replace, markdown)


def normalize_hackmd_markdown(markdown: str) -> str:
    """Normalize image HTML while preserving headings, tables and source order."""
    markdown = markdown.replace("\r\n", "\n")
    return _reference_images_to_inline(_html_images_to_markdown(markdown))


def _load_image(url: str):
    from PIL import Image, UnidentifiedImageError

    payload, content_type, _ = _fetch(
        url,
        limit=MAX_IMAGE_BYTES,
        validator=_validate_public_image_url,
    )
    if content_type == "image/svg+xml" or urlsplit(url).path.lower().endswith(".svg"):
        raise HackMDImportError("目前不支援 SVG 圖片，請先轉成 PNG、JPG 或 WebP。")
    try:
        with Image.open(BytesIO(payload)) as source:
            source.load()
            if source.width * source.height > 80_000_000:
                raise HackMDImportError("圖片像素數過大，已停止處理。")
            if source.mode not in {"RGB", "RGBA"}:
                return source.convert("RGBA" if "transparency" in source.info else "RGB")
            return source.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise HackMDImportError(f"下載內容不是可辨識圖片：{url[:160]}") from exc


def _image_map(markdown: str, note_url: str) -> tuple[dict[str, object], list[dict]]:
    references: list[tuple[str, str]] = []
    seen_targets: set[str] = set()
    for match in IMAGE_PATTERN.finditer(markdown):
        raw_target = html.unescape(match.group("target").strip("<>"))
        if raw_target in seen_targets:
            continue
        seen_targets.add(raw_target)
        references.append((raw_target, urljoin(note_url, raw_target)))
    if len(references) > MAX_IMAGES:
        raise HackMDImportError(
            f"筆記包含 {len(references)} 張圖片，超過單次 {MAX_IMAGES} 張限制。"
        )

    images: dict[str, object] = {}
    downloads: list[dict] = []
    cache: dict[str, object] = {}
    failures: list[str] = []
    for raw_target, resolved_url in references:
        try:
            image = cache.get(resolved_url)
            if image is None:
                image = _load_image(resolved_url)
                cache[resolved_url] = image
            images[raw_target] = image
            downloads.append({
                "source": raw_target,
                "resolved_url": resolved_url,
                "status": "downloaded",
                "pixel_size": list(getattr(image, "size", ())),
            })
        except HackMDImportError as exc:
            failures.append(f"{raw_target}: {exc}")
            downloads.append({
                "source": raw_target,
                "resolved_url": resolved_url,
                "status": "failed",
                "error": str(exc),
            })
    if failures:
        detail = "；".join(failures[:8])
        if len(failures) > 8:
            detail += f"；另有 {len(failures) - 8} 張"
        raise HackMDImportError(f"部分圖片無法安全下載，未建立不完整文件：{detail}")
    return images, downloads


def _document_title(markdown: str, note_url: str) -> str:
    front_matter = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", markdown, re.S)
    if front_matter:
        title = re.search(r"(?m)^title:\s*[\"']?(.+?)[\"']?\s*$", front_matter.group(1))
        if title:
            return title.group(1).strip()
    heading = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    if heading:
        return heading.group(1).strip()
    return unquote(urlsplit(note_url).path.rstrip("/").split("/")[-1]) or "hackmd-note"


def _safe_stem(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", value).strip(" ._")
    return value[:80] or "hackmd-note"


def prepare_hackmd_markdown(
    note_url: str,
    output_dir: str | Path,
    *,
    document_type: str,
    image_provider: str | None = None,
    image_model: str | None = None,
) -> tuple[Path, str, dict]:
    raw_markdown, response_url = download_hackmd_markdown(note_url)
    markdown = normalize_hackmd_markdown(raw_markdown)
    # This zero-token gate deliberately runs before image downloads and all
    # multimodal/LLM calls, so unrelated notes cannot consume captioning tokens.
    relevance = validate_project_document_relevance(markdown, document_type)
    images, downloads = _image_map(markdown, note_url)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = _safe_stem(_document_title(markdown, note_url))
    markdown_path = destination / f"{timestamp}_{stem}.md"
    source_path = destination / f"{timestamp}_{stem}_source.md"
    assets_dir = destination / f"{markdown_path.stem}_assets"
    manifest_path = destination / f"{markdown_path.stem}_images.json"
    source_path.write_text(raw_markdown, encoding="utf-8")

    settings = ImageCaptionSettings.from_config(
        enabled=True,
        provider=image_provider,
        model=image_model,
        fail_on_error=False,
    )
    # HackMD imports promise per-image processing. Do not silently skip icons or
    # screenshots because of the generic PDF minimum-size setting.
    settings = replace(
        settings,
        max_images=MAX_IMAGES,
        min_width=1,
        min_height=1,
    )
    enriched, manifest = enrich_markdown_with_marker_images(
        markdown,
        images,
        assets_dir=assets_dir,
        markdown_assets_path=assets_dir.name,
        settings=settings,
        captioner=ExternalMultimodalCaptioner(
            settings, document_type=document_type
        ),
        manifest_path=manifest_path,
        document_type=document_type,
    )
    manifest.update({
        "schema_version": max(4, int(manifest.get("schema_version", 0))),
        "conversion_source": "hackmd_public_markdown",
        "source_url": note_url,
        "download_url": response_url,
        "source_sha256": hashlib.sha256(raw_markdown.encode("utf-8")).hexdigest(),
        "image_references": downloads,
        "project_relevance": relevance.to_dict(),
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(enriched, encoding="utf-8")
    return markdown_path, enriched, manifest
