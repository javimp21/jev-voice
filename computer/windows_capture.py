"""Ephemeral foreground-window capture with explicit screen-coordinate metadata."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from computer.models import CaptureDiagnostics, Rect, ScreenshotMetadata, VisualElement
from computer.visual import ScreenshotCapture
from computer.visual_providers.common import png_bytes


class CaptureUnavailable(RuntimeError):
    """Foreground pixels cannot be captured with trustworthy geometry."""


def _input_desktop_is_default() -> bool:
    """Fail closed outside the ordinary interactive desktop."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.OpenInputDesktop.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    user32.OpenInputDesktop.restype = wintypes.HANDLE
    user32.CloseDesktop.argtypes = (wintypes.HANDLE,)
    user32.CloseDesktop.restype = wintypes.BOOL
    user32.GetUserObjectInformationW.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    user32.GetUserObjectInformationW.restype = wintypes.BOOL
    desktop = user32.OpenInputDesktop(0, False, 0x0001)
    if not desktop:
        return False
    try:
        needed = wintypes.DWORD(0)
        user32.GetUserObjectInformationW(desktop, 2, None, 0, ctypes.byref(needed))
        if needed.value <= 2 or needed.value > 1024:
            return False
        buffer = ctypes.create_unicode_buffer(needed.value // 2)
        if not user32.GetUserObjectInformationW(
            desktop, 2, buffer, ctypes.sizeof(buffer), ctypes.byref(needed),
        ):
            return False
        return buffer.value.casefold() == "default"
    finally:
        user32.CloseDesktop(desktop)


def _window_rect(handle: int) -> Rect:
    import win32gui
    if not handle or not win32gui.IsWindow(handle) or win32gui.IsIconic(handle):
        raise CaptureUnavailable("Foreground window is unavailable or minimized.")
    left, top, right, bottom = win32gui.GetWindowRect(handle)
    if right <= left or bottom <= top:
        raise CaptureUnavailable("Foreground window has invalid bounds.")
    return Rect(left, top, right, bottom)


def _virtual_screen() -> Rect:
    import win32api
    return Rect(
        win32api.GetSystemMetrics(76), win32api.GetSystemMetrics(77),
        win32api.GetSystemMetrics(76) + win32api.GetSystemMetrics(78),
        win32api.GetSystemMetrics(77) + win32api.GetSystemMetrics(79),
    )


def _intersection(first: Rect, second: Rect) -> Rect:
    result = Rect(max(first.left, second.left), max(first.top, second.top),
                  min(first.right, second.right), min(first.bottom, second.bottom))
    if result.right <= result.left or result.bottom <= result.top:
        raise CaptureUnavailable("Foreground window is outside the visible virtual screen.")
    return result


def _dpi(handle: int) -> tuple[int, int]:
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetDpiForWindow.argtypes = (wintypes.HWND,)
        user32.GetDpiForWindow.restype = wintypes.UINT
        dpi = int(user32.GetDpiForWindow(handle))
        return (dpi, dpi) if dpi > 0 else (96, 96)
    except Exception:
        return 96, 96


def _grab(bounds: Rect) -> Any:
    from PIL import ImageGrab
    return ImageGrab.grab(
        bbox=(bounds.left, bounds.top, bounds.right, bounds.bottom), all_screens=True,
    )


def _mask(image: Any, regions: Sequence[Rect], metadata: ScreenshotMetadata) -> tuple[int, int]:
    if not regions:
        return 0, 0
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    count = 0
    area = 0
    for region in regions:
        left = max(0, round((region.left - metadata.capture_bounds.left) * metadata.scale_x))
        top = max(0, round((region.top - metadata.capture_bounds.top) * metadata.scale_y))
        right = min(metadata.pixel_width, round((region.right - metadata.capture_bounds.left) * metadata.scale_x))
        bottom = min(metadata.pixel_height, round((region.bottom - metadata.capture_bounds.top) * metadata.scale_y))
        if right > left and bottom > top:
            draw.rectangle((left, top, right - 1, bottom - 1), fill="black")
            count += 1
            area += (right - left) * (bottom - top)
    return count, min(area, metadata.pixel_width * metadata.pixel_height)


def _content_metrics(image: Any) -> tuple[int, tuple[int, int, int], tuple[int, int, int], float, float, float, float]:
    """Bounded image statistics; never OCR or retain sampled pixels."""
    from PIL import Image, ImageStat
    sample = image.convert("RGB")
    try:
        sample.thumbnail((128, 128), Image.Resampling.BILINEAR)
        pixels = tuple(sample.get_flattened_data())
        extrema = ImageStat.Stat(sample).extrema
        luminance = tuple(.2126 * r + .7152 * g + .0722 * b for r, g, b in pixels)
        count = max(1, len(luminance))
        mean = sum(luminance) / count
        variance = sum((value - mean) ** 2 for value in luminance) / count
        return (
            len(set(pixels)),
            tuple(int(item[0]) for item in extrema),
            tuple(int(item[1]) for item in extrema),
            round(mean, 3), round(math.sqrt(variance), 3),
            round(100 * sum(value <= 10 for value in luminance) / count, 3),
            round(100 * sum(value >= 245 for value in luminance) / count, 3),
        )
    finally:
        sample.close()


class WindowsWindowCapture:
    """Capture only the visible portion of the exact foreground window."""

    _SECURE_PROCESSES = frozenset({"consent.exe", "logonui.exe", "winlogon.exe"})

    def __init__(
        self, *, grabber: Callable[[Rect], Any] = _grab,
        desktop_check: Callable[[], bool] = _input_desktop_is_default,
    ) -> None:
        self._grabber = grabber
        self._desktop_check = desktop_check

    def current_window_bounds(self, handle: int) -> Rect:
        return _window_rect(handle)

    def current_virtual_screen_bounds(self) -> Rect:
        return _virtual_screen()

    def capture(
        self, snapshot_id: str, expected_handle: int, process_name: str,
        sensitive_regions: Sequence[Rect] = (),
    ) -> ScreenshotCapture:
        if sys.platform != "win32" or not self._desktop_check():
            raise CaptureUnavailable("Screen capture is unavailable outside the default desktop.")
        if Path(process_name).name.casefold() in self._SECURE_PROCESSES:
            raise CaptureUnavailable("Secure desktop windows cannot be captured.")
        import win32gui
        if win32gui.GetForegroundWindow() != expected_handle:
            raise CaptureUnavailable("Foreground window changed before capture.")
        window_bounds = _window_rect(expected_handle)
        virtual_bounds = _virtual_screen()
        capture_bounds = _intersection(window_bounds, virtual_bounds)
        image = self._grabber(capture_bounds)
        width, height = getattr(image, "size", (0, 0))
        screen_width = capture_bounds.right - capture_bounds.left
        screen_height = capture_bounds.bottom - capture_bounds.top
        if (not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1
                or screen_width < 1 or screen_height < 1):
            close = getattr(image, "close", None)
            if callable(close):
                close()
            raise CaptureUnavailable("Screenshot dimensions are invalid.")
        scale_x, scale_y = width / screen_width, height / screen_height
        if not all(math.isfinite(value) and 0.25 <= value <= 8 for value in (scale_x, scale_y)):
            raise CaptureUnavailable("Screenshot coordinate scale is untrustworthy.")
        dpi_x, dpi_y = _dpi(expected_handle)
        metadata = ScreenshotMetadata(
            snapshot_id, expected_handle, window_bounds, capture_bounds,
            width, height, dpi_x, dpi_y, scale_x, scale_y,
        )
        masked, masked_area = _mask(image, sensitive_regions, metadata)
        if masked:
            metadata = ScreenshotMetadata(
                snapshot_id, expected_handle, window_bounds, capture_bounds,
                width, height, dpi_x, dpi_y, scale_x, scale_y, masked,
            )
        capture = ScreenshotCapture(metadata, image)
        encoded = png_bytes(capture)
        unique, channel_min, channel_max, mean, stddev, black, white = _content_metrics(image)
        try:
            import win32process
            _thread, foreground_pid = win32process.GetWindowThreadProcessId(expected_handle)
            foreground_pid = int(foreground_pid) if foreground_pid else None
        except Exception:
            foreground_pid = None
        capture.diagnostics = CaptureDiagnostics(
            expected_handle, foreground_pid, window_bounds, capture_bounds, virtual_bounds,
            width, height, len(encoded), masked,
            round(100 * masked_area / (width * height), 3),
            "pillow_imagegrab_screen_region",
            bool(getattr(win32gui, "IsWindowVisible", lambda _handle: True)(expected_handle)),
            bool(getattr(win32gui, "IsIconic", lambda _handle: False)(expected_handle)),
            unique, channel_min, channel_max,
            mean, stddev, black, white,
        )
        return capture


def save_debug_screenshot(capture: ScreenshotCapture, path: Path) -> None:
    """Explicit, no-overwrite PNG export for local geometry debugging."""
    if path.suffix.casefold() != ".png":
        raise ValueError("Debug screenshot path must end in .png")
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError("Debug screenshot path must be new and its parent must exist.")
    with path.open("xb") as output:
        output.write(png_bytes(capture))


def save_debug_overlay(
    capture: ScreenshotCapture, elements: Sequence[VisualElement], path: Path,
) -> None:
    """Explicit no-overwrite visual grounding overlay in capture-pixel coordinates."""
    if path.suffix.casefold() != ".png":
        raise ValueError("Debug overlay path must end in .png")
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError("Debug overlay path must be new and its parent must exist.")
    from PIL import ImageDraw
    image = capture.image.copy()
    try:
        draw = ImageDraw.Draw(image)
        for element in elements:
            rect = element.rectangle
            color = "#00ff66"
            draw.rectangle((rect.left, rect.top, rect.right - 1, rect.bottom - 1), outline=color, width=3)
            label = f"{element.id} {element.label[:32]}".strip()
            text_box = draw.textbbox((rect.left, rect.top), label)
            height = text_box[3] - text_box[1] + 4
            top = max(0, rect.top - height)
            draw.rectangle((rect.left, top, min(image.width, text_box[2] + 4), rect.top), fill="black")
            draw.text((rect.left + 2, top + 1), label, fill=color)
        with path.open("xb") as output:
            image.save(output, format="PNG")
    finally:
        image.close()
