#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""General-purpose CLI for OpenAI GPT Image 2.

Mirrors the two official endpoints from the OpenAI cookbook using an
OpenAI-compatible HTTP API:

    client.images.generate(...)   — text → image          (no  -i)
    client.images.edit(...)       — text + image(s) → image (with -i; mask via -m)

Every documented parameter is exposed as a flag. Reads OPENAI_API_KEY or
ANTHROPIC_AUTH_TOKEN from process env, then .env, then ~/.env without overriding
existing env. Writes the returned PNG/JPEG/WebP bytes to disk and prints the
output path(s) on stdout.

Exit codes: 0 success, 1 API error, 2 bad args.

Examples:
    # Basic generate, auto filename, 1K square
    gpt-image -p "a cat astronaut on the moon"

    # Named output, portrait 2K, high quality
    gpt-image -p "Chinese tea poster" -f poster.png --size 2k --quality high

    # Edit existing image (colorize, restyle, translate text, etc.)
    gpt-image -p "colorize this manga page" -i page.jpg -f colored.png

    # Multi-reference edit (outfit transfer, pet + brand, etc.)
    gpt-image -p "77 × KFC collab poster" -i cat.png -i kfc_logo.png -f collab.png

    # Alpha-channel inpaint (mask opaque = keep, transparent = regenerate)
    gpt-image -p "replace sky with aurora" -i photo.jpg -m sky_mask.png -f aurora.png

    # Grid of 4, transparent background, webp
    gpt-image -p "isometric chair, minimalist" -n 4 --background opaque --format webp

    # Skill launcher (same implementation, installed skill-folder path)
    uv run "$SKILL_DIR/scripts/generate.py" -p "a cat astronaut on the moon"
"""
from __future__ import annotations

import argparse
import base64
import binascii
import ipaddress
import json
import mimetypes
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


CREDENTIAL_ENV_KEYS = {"OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
BASE_URL_ENV_KEYS = {"OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"}


def _read_env_file(path: Path, allowed_keys: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in allowed_keys:
            values[key] = value.strip().strip('"').strip("'")
    return values


def _load_env_chain() -> dict[str, str]:
    """Resolve API credentials without overriding runtime-provided env.

    Order: process env → ./.env → ~/.env. Existing process env wins so
    hosted agents or explicit shell exports are not replaced by local files.
    """
    initial_env = {key: value for key, value in os.environ.items() if key in CREDENTIAL_ENV_KEYS | BASE_URL_ENV_KEYS}
    cwd_values = _read_env_file(Path.cwd() / ".env", CREDENTIAL_ENV_KEYS)
    home_values = _read_env_file(Path.home() / ".env", CREDENTIAL_ENV_KEYS | BASE_URL_ENV_KEYS)
    for key, value in (home_values | cwd_values).items():
        if key not in os.environ:
            os.environ[key] = value
    return initial_env


SIZE_SHORTCUTS: dict[str, str] = {
    "1k": "1024x1024",
    "2k": "2048x2048",
    "4k": "3840x2160",
    "portrait": "1024x1536",
    "landscape": "1536x1024",
    "square": "1024x1024",
    "wide": "2048x1152",
    "tall": "2160x3840",
}

DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "1024x1024"
DEFAULT_MODERATION = "low"


def slugify(text: str, max_len: int = 30) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "-", s)[:max_len]
    return s or "image"


def default_output_path(prompt: str, extension: str) -> Path:
    cwd = Path.cwd()
    target_dir = cwd / "fig" if (cwd / "fig").is_dir() else cwd
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return target_dir / f"{stamp}-{slugify(prompt)}.{extension}"


def resolve_size(value: str) -> str:
    return SIZE_SHORTCUTS.get(value.lower(), value)


def model_rejects_input_fidelity(model: str) -> bool:
    return model.strip().lower().startswith("gpt-image-2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gpt-image",
        description="Call OpenAI GPT Image 2 (generations or edits) via an OpenAI-compatible HTTP API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-p", "--prompt", required=True, help="Text prompt / edit instruction.")
    p.add_argument(
        "-f", "--file",
        help="Output path. Auto-generated as YYYY-MM-DD-HH-MM-SS-<slug>.<ext> if omitted "
             "(written to ./fig/ if that dir exists, else ./).",
    )
    p.add_argument(
        "-i", "--image", action="append", type=Path, default=None,
        help="Reference image path. Repeat flag for multi-reference edits. "
             "Presence of any -i switches endpoint to client.images.edit().",
    )
    p.add_argument(
        "-m", "--mask", type=Path, default=None,
        help="Alpha-channel PNG mask (opaque = preserved, transparent = regenerated). "
             "Edits endpoint only; requires -i.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default {DEFAULT_MODEL}).")
    p.add_argument(
        "--size", default=DEFAULT_SIZE,
        help="Image size. Accepts literals (1024x1024, 1536x1024, 2048x2048, 3840x2160, "
             "any 16px-multiple up to 3840 max edge, 3:1 ratio cap) or shortcuts "
             "(1k, 2k, 4k, portrait, landscape, square, wide, tall). Default 1024x1024.",
    )
    p.add_argument(
        "--quality", default="high", choices=["auto", "low", "medium", "high"],
        help="Rendering fidelity / budget knob (cost scales ~10× per step). Default high. "
             "Use low for cheap drafts, medium for normal exploration, high for final text-heavy or shipping-facing assets.",
    )
    p.add_argument("-n", "--n", type=int, default=1, help="Number of images to return. Default 1.")
    p.add_argument(
        "--background", default=None, choices=["auto", "opaque"],
        help="`opaque` disables transparency. Default API-side auto.",
    )
    p.add_argument(
        "--moderation", default=DEFAULT_MODERATION, choices=["auto", "low"],
        help="Generations only. Default low. Use `auto` if you want the stricter API-side default.",
    )
    p.add_argument(
        "--input-fidelity", dest="input_fidelity", default=None, choices=["low", "high"],
        help="Edits only. gpt-image-2 rejects this parameter, so the CLI drops it locally before calling the API.",
    )
    p.add_argument(
        "--format", dest="output_format", default=None,
        choices=["png", "jpeg", "webp"],
        help="Output encoding. Default png.",
    )
    p.add_argument(
        "--compression", dest="output_compression", type=int, default=None,
        help="0-100 compression level for jpeg/webp. Ignored for png.",
    )
    p.add_argument(
        "--user", default=None,
        help="Optional end-user identifier forwarded to OpenAI for abuse tracking.",
    )
    return p.parse_args()


def _filter_none(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None."""
    return {k: v for k, v in d.items() if v is not None}


def _hostname_is_local(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except socket.gaierror:
        addresses = {hostname}
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
    return False


def _validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("base URL must not contain embedded credentials")
    normalized = value.rstrip("/")
    if not urllib.parse.urlparse(normalized).path.rstrip("/").endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _validate_https_url(value: str, label: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"{label} must be an https URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not contain embedded credentials")
    if _hostname_is_local(parsed.hostname):
        raise ValueError(f"{label} must not point to a local or private network address")
    return value.rstrip("/")


class ApiRequestError(Exception):
    pass


def _redact_secrets(text: str) -> str:
    redacted = text
    for key in CREDENTIAL_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            redacted = redacted.replace(value, "[REDACTED]").replace(f"Bearer {value}", "Bearer [REDACTED]")
    return redacted


def _resolve_api_config(initial_env: dict[str, str]) -> tuple[str | None, str | None, str | None]:
    if initial_env.get("OPENAI_API_KEY"):
        return initial_env["OPENAI_API_KEY"], initial_env.get("OPENAI_BASE_URL"), "openai"
    if initial_env.get("ANTHROPIC_AUTH_TOKEN"):
        return initial_env["ANTHROPIC_AUTH_TOKEN"], initial_env.get("ANTHROPIC_BASE_URL"), "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], os.environ.get("OPENAI_BASE_URL"), "openai"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return os.environ["ANTHROPIC_AUTH_TOKEN"], os.environ.get("ANTHROPIC_BASE_URL"), "anthropic"
    return None, None, None


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.base_url = _validate_base_url(base_url or "https://api.openai.com/v1")
        self.api_key = api_key
        self.images = ImageApi(self)

    @property
    def auth_header(self) -> str:
        if self.api_key.lower().startswith("bearer "):
            return self.api_key
        return f"Bearer {self.api_key}"

    def request_json(self, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(_filter_none(payload)).encode("utf-8"),
            headers={
                "Authorization": self.auth_header,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return _open_json(request)

    def request_multipart(self, path: str, fields: dict[str, Any], files: list[tuple[str, Any]]) -> Any:
        boundary = "----gptimageboundary"
        body = _build_multipart_body(boundary, fields, files)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": self.auth_header,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        return _open_json(request)


def _open_json(request: urllib.request.Request) -> Any:
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = _redact_secrets(e.read().decode("utf-8", errors="replace"))
        raise ApiRequestError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise ApiRequestError(_redact_secrets(str(e.reason))) from e


def _api_response(payload: Any) -> Any:
    data = payload.get("data") if isinstance(payload, dict) else None
    return SimpleNamespace(data=[SimpleNamespace(**item) for item in data or []])


def _build_multipart_body(boundary: str, fields: dict[str, Any], files: list[tuple[str, Any]]) -> bytes:
    parts: list[bytes] = []
    for name, value in _filter_none(fields).items():
        parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
            f"{value}\r\n".encode("utf-8"),
        ])
    for name, handle in files:
        filename = Path(handle.name).name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            handle.read(),
            b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts)


class ImageApi:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def generate(self, **payload: Any) -> Any:
        return _api_response(self.client.request_json("/images/generations", payload))

    def edit(self, **payload: Any) -> Any:
        images = payload.pop("image")
        mask = payload.pop("mask", None)
        files = [("image" if len(images) == 1 else "image[]", image) for image in images]
        if mask:
            files.append(("mask", mask))
        return _api_response(self.client.request_multipart("/images/edits", payload, files))


def call_generate(client: OpenAICompatibleClient, args: argparse.Namespace) -> Any:
    return client.images.generate(**_filter_none({
        "model": args.model,
        "prompt": args.prompt,
        "size": resolve_size(args.size),
        "quality": args.quality,
        "n": args.n,
        "background": args.background,
        "moderation": args.moderation,
        "output_format": args.output_format,
        "output_compression": args.output_compression,
        "user": args.user,
    }))


def call_edit(client: OpenAICompatibleClient, args: argparse.Namespace) -> Any:
    for p in args.image:
        if not p.is_file():
            print(f"error: --image not found: {p}", file=sys.stderr)
            sys.exit(2)
    if args.mask and not args.mask.is_file():
        print(f"error: --mask not found: {args.mask}", file=sys.stderr)
        sys.exit(2)

    input_fidelity = args.input_fidelity
    if input_fidelity and model_rejects_input_fidelity(args.model):
        print(
            "note: dropping --input-fidelity because gpt-image-2 rejects that parameter.",
            file=sys.stderr,
        )
        input_fidelity = None

    image_handles = [p.open("rb") for p in args.image]
    mask_handle = args.mask.open("rb") if args.mask else None
    try:
        return client.images.edit(**_filter_none({
            "model": args.model,
            "image": image_handles,
            "mask": mask_handle,
            "prompt": args.prompt,
            "size": resolve_size(args.size),
            "quality": args.quality,
            "n": args.n,
            "background": args.background,
            "input_fidelity": input_fidelity,
            "output_format": args.output_format,
            "output_compression": args.output_compression,
            "user": args.user,
        }))
    finally:
        for h in image_handles:
            h.close()
        if mask_handle:
            mask_handle.close()


def write_outputs(data: list[Any], out_path: Path, n: int) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, item in enumerate(data):
        b64 = getattr(item, "b64_json", None)
        if b64:
            raw = base64.b64decode(b64)
        else:
            print(f"error: response item {i} has no b64_json image data", file=sys.stderr)
            sys.exit(1)

        if n == 1:
            target = out_path
        else:
            stem = out_path.with_suffix("")
            target = stem.parent / f"{stem.name}_{i}{out_path.suffix}"
        target.write_bytes(raw)
        written.append(target)
    return written


def main() -> int:
    args = parse_args()

    initial_env = _load_env_chain()
    api_key, base_url, provider = _resolve_api_config(initial_env)
    if not api_key:
        print(
            "error: OPENAI_API_KEY not set and ANTHROPIC_AUTH_TOKEN fallback not available. "
            "Add one to env / .env / ~/.env, or use your host agent's native image tool.",
            file=sys.stderr,
        )
        return 2
    if provider == "anthropic" and not base_url:
        print(
            "error: ANTHROPIC_AUTH_TOKEN fallback requires ANTHROPIC_BASE_URL pointing to an OpenAI-compatible endpoint.",
            file=sys.stderr,
        )
        return 2

    if args.mask and not args.image:
        print("error: --mask requires --image (edits endpoint only)", file=sys.stderr)
        return 2

    ext = args.output_format or "png"
    out_path = Path(args.file).expanduser().resolve() if args.file else default_output_path(args.prompt, ext)

    try:
        client = OpenAICompatibleClient(api_key=api_key, base_url=base_url)
        result = call_edit(client, args) if args.image else call_generate(client, args)
    except (ApiRequestError, ValueError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    data = result.data or []
    if not data:
        print(f"error: no image data in response: {result}", file=sys.stderr)
        return 1

    try:
        for p in write_outputs(data, out_path, args.n):
            print(p)
    except (binascii.Error, OSError, ValueError, urllib.error.URLError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
