import win32gui
import win32ui
from ctypes import windll
from PIL import Image
import time
import os
import configparser
from discord_webhook import DiscordWebhook
import pytesseract
from diffimg import diff
from loguru import logger
import sys
import re
from PIL import ImageEnhance, ImageFilter, ImageOps, ImageDraw
from typing import List, Optional
from pathlib import Path
import hashlib
import json
import difflib

# Simple ARK bot

# Setup logging like original
ENABLE_DEBUG = True  # Set to True to enable debug logs

LOG_LEVEL = "DEBUG" if ENABLE_DEBUG else "INFO"
logger.remove()
logger.add(
    sys.stderr,
    format="<g>{time:HH:mm:ss}</g> <lvl>{message}</lvl>",
    level=LOG_LEVEL,
)

who = "WindowLickers"
# Dynamically detect 'who' from map area OCR once per run
WHO_LOCKED = False  # Set to True after successful OCR assignment
# Base coordinates for the map area that contains the desired label/name
MAP_AREA_BASE_RECT = (350, 334, 524, 378)  # (left, top, right, bottom)

# Set DPI awareness
# Last OCR text for TRIBES header (for debug prints)
LAST_TRIBES_OCR: str = ""

# Debug screenshots directory and retention
DEBUG_DIR = Path("debug_screens")
MAX_DEBUG_FILES = int(os.getenv("DEBUG_MAX_FILES", "500"))
SAVE_FULL_LOGS_OCR = os.getenv("DEBUG_SAVE_FULL_LOGS_OCR", "0") in ("1", "true", "True")

# Rolling file paths (kept inside debug folder, not project root)
ROLLING_SCREENSHOT = DEBUG_DIR / "screenshot.png"
LOG_NEW_PATH = DEBUG_DIR / "log_new.png"
LOG_OLD_PATH = DEBUG_DIR / "log_old.png"
TRIBES_CHECK_PATH = DEBUG_DIR / "tribes_check.png"
FULL_LOGS_OCR_PATH = DEBUG_DIR / "full_logs_ocr.png"
HEADER_PATH = DEBUG_DIR / "log_header.png"

# Minimum change threshold to prevent spam
MIN_CHANGE_THRESHOLD = 0.020  # Increased from 0.010
CHANGE_COOLDOWN = 5  # Seconds to wait before processing another change


def _ensure_debug_dir():
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create debug dir {DEBUG_DIR}: {e}")


def _prune_debug_dir():
    try:
        files = sorted(DEBUG_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
        excess = max(0, len(files) - MAX_DEBUG_FILES)
        for p in files[:excess]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Prune debug dir failed: {e}")


def save_debug_image(
    img: Image.Image, base_name: str, keep_only_latest: bool = True
) -> None:
    """Save PNGs in debug folder.
    - If keep_only_latest=True: save as '<base_name>.png' (single rolling file).
    - If keep_only_latest=False: save timestamped '<base_name>_YYYYMMDD-HHMMSS.png'.
    Also prunes old timestamped files according to MAX_DEBUG_FILES.
    """
    try:
        _ensure_debug_dir()
        if keep_only_latest:
            # Clean up any old timestamped variants for this base
            for p in DEBUG_DIR.glob(f"{base_name}_*.png"):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            out = DEBUG_DIR / f"{base_name}.png"
        else:
            ts = time.strftime("%Y%m%d-%H%M%S")
            out = DEBUG_DIR / f"{base_name}_{ts}.png"
        img.save(out)
        _prune_debug_dir()
    except Exception as e:
        logger.debug(f"save_debug_image failed for {base_name}: {e}")


# Persistent state for deduplication
STATE_FILE = DEBUG_DIR / "state.json"


def _load_state() -> dict:
    # Ensure state has keys we use
    base = {
        "last_processed_line": "",
        "last_seen_ts": 0,
        "last_ingame_ts": 0,
        "last_header": "",
        "last_header_ts": 0,
        "last_change_time": 0,  # Add cooldown tracking
        "last_content_hash": "",  # Add content hash for better deduplication
    }
    try:
        if STATE_FILE.exists():
            merged = {**base, **json.loads(STATE_FILE.read_text(encoding="utf-8"))}
            return merged
    except Exception as e:
        logger.debug(f"load_state failed: {e}")
    return base


def _save_state(state: dict) -> None:
    try:
        _ensure_debug_dir()
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.debug(f"save_state failed: {e}")


def _is_new_processed_line(line: str) -> bool:
    state = _load_state()
    last = state.get("last_processed_line", "")

    if not line or len(line.strip()) < 10:  # Ignore very short lines
        return False

    # Use normalized content for duplicate suppression
    norm = re.sub(r"\s+", " ", line.strip())
    # Remove Discord timestamp formatting for comparison
    norm_content = re.sub(r"<t:\d+:[TF]>\s*", "", norm)

    last_norm = re.sub(r"\s+", " ", last.strip())
    last_norm_content = re.sub(r"<t:\d+:[TF]>\s*", "", last_norm)

    if norm_content != last_norm_content and len(norm_content) > 5:
        state["last_processed_line"] = line
        state["last_seen_ts"] = int(time.time())
        _save_state(state)
        return True
    return False


def _get_last_ingame_ts() -> int:
    state = _load_state()
    return int(state.get("last_ingame_ts", 0) or 0)


def _update_last_ingame_ts(v: int) -> None:
    state = _load_state()
    state["last_ingame_ts"] = int(v)
    _save_state(state)


def _is_on_cooldown() -> bool:
    """Check if we're still in cooldown period from last change"""
    state = _load_state()
    last_change = state.get("last_change_time", 0)
    return (time.time() - last_change) < CHANGE_COOLDOWN


def _update_last_change_time() -> None:
    """Update the last change time to now"""
    state = _load_state()
    state["last_change_time"] = time.time()
    _save_state(state)


# Load config and environment with fallbacks
config = configparser.ConfigParser()
config.read("Tribelog bot.ini")

# Optional OpenCV for template matching (improves UI detection if available)
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# Tesseract configuration: honor ini overrides or bundled copy
from pytesseract import pytesseract as _pt


def _configure_tesseract():
    try:
        ini_cmd = config.get("Tesseract", "cmd", fallback="").strip()
        candidates = []
        if ini_cmd:
            candidates.append(ini_cmd)
        # Try local bundled copy if present
        candidates.append(str(Path("Tesseract") / "tesseract.exe"))
        for cmd in candidates:
            if cmd and Path(cmd).exists():
                _pt.tesseract_cmd = cmd
                # Set tessdata prefix if bundled data exists
                tessdata_dir = Path(cmd).parent / "tessdata"
                if tessdata_dir.exists():
                    os.environ.setdefault("TESSDATA_PREFIX", str(tessdata_dir))
                logger.debug(f"Using Tesseract at: {cmd}")
                break
        else:
            logger.debug("Using system Tesseract (no local/bundled cmd found)")
    except Exception as e:
        logger.debug(f"_configure_tesseract failed: {e}")


_configure_tesseract()

# Template directory (drop assets from external repo here)
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", "overrides/templates"))

# Read template-based alignment config
ANCHOR_FILE = config.get(
    "Templates", "anchor_file", fallback=str(TEMPLATES_DIR / "tribes_anchor.png")
).strip()
ANCHOR_BASE_X = config.getint("Templates", "anchor_base_x", fallback=176)
ANCHOR_BASE_Y = config.getint("Templates", "anchor_base_y", fallback=115)
ANCHOR_MIN_SCORE = float(config.get("Templates", "anchor_min_score", fallback="0.75"))
ANCHOR_SEARCH_MARGIN = config.getint("Templates", "anchor_search_margin", fallback=400)


def _split_webhook_urls(raw: str) -> List[str]:
    if not raw:
        return []
    # Support comma, semicolon, whitespace and newlines
    parts = [p.strip() for p in re.split(r"[\s,;]+", raw) if p.strip()]
    # Basic validation: must start with http(s)
    return [p for p in parts if p.lower().startswith("http")]


# Prefer environment variable if provided
alert_urls_raw = os.getenv("DISCORD_WEBHOOK_URLS") or config.get(
    "Log_Alert", "alert_urls", fallback=""
)
roles = os.getenv("DISCORD_ROLES") or config.get("Role", "roles", fallback="")

# Format role mention(s) like <@&ID>
ROLE_MENTION = ""
try:
    ids = [r.strip() for r in re.split(r"[\s,;]+", roles) if r.strip()]
    mentions = [f"<@&{rid}>" for rid in ids if rid.isdigit()]
    ROLE_MENTION = " ".join(mentions)
except Exception:
    ROLE_MENTION = roles  # fallback to raw

# Store list of webhook endpoints
ALERT_WEBHOOK_URLS: List[str] = _split_webhook_urls(alert_urls_raw)

# --- Game open? ---


def _find_ark_window():
    # Try exact and fuzzy matches for the Ark window title
    candidates = []
    exact_titles = ["ArkAscended", "ARK: Survival Ascended"]

    for title in exact_titles:
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            candidates.append((hwnd, title))

    if not candidates:
        # Fallback: enumerate windows and find any that contain Ark keywords
        def _enum_handler(h, _):
            try:
                t = win32gui.GetWindowText(h)
                if t and any(k in t.lower() for k in ["ark", "ascended"]):
                    candidates.append((h, t))
            except Exception:
                pass

        win32gui.EnumWindows(_enum_handler, None)


# --- Coordinate helpers (no resolution scaling) ---
# All coordinates are treated as absolute pixels relative to the captured image/window.


# --- Coordinate helpers (scaled to the actual captured image size) ---
# Base coordinates are authored for 1920x1080; we scale to the current image size at runtime.
# Resolution auto-detection: scaling decided by Ark window size in _scale().


def _scale(img: Image.Image) -> tuple[float, float]:
    """Return scaling factors auto-detected from the Ark window or image size.
    - If RES_MODE is explicitly FHD or QHD (via env/ini), use that.
    - Otherwise, detect by width/height: QHD if width>=2400 or height>=1400, else FHD.
    Returns (sx, sy) with the same factor for both axes.
    """
    # Auto-detect based on the Ark window rect or the screenshot image
    try:
        if img is not None:
            w, h = img.width, img.height
        else:
            hwnd = _find_ark_window()
            if hwnd:
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                w = max(0, right - left)
                h = max(0, bottom - top)
            else:
                w = h = 0
        # Heuristic thresholds for QHD vs FHD using window size
        if w >= 2400 or h >= 1400:
            return 4.0 / 3.0, 4.0 / 3.0  # QHD
        else:
            return 1.0, 1.0  # FHD
    except Exception:
        return 1.0, 1.0


# UI offset (pixels in current image coords), updated via template matching if available
UI_OFFSET_PIXELS = [0, 0]


def srect(
    img: Image.Image, l: int, t: int, r: int, b: int
) -> tuple[int, int, int, int]:
    sx, sy = _scale(img)
    L = int(round(l * sx))
    T = int(round(t * sy))
    R = int(round(r * sx))
    B = int(round(b * sy))
    # Apply detected UI offset
    L += UI_OFFSET_PIXELS[0]
    R += UI_OFFSET_PIXELS[0]
    T += UI_OFFSET_PIXELS[1]
    B += UI_OFFSET_PIXELS[1]
    # Clamp to image bounds
    L = max(0, min(L, img.width))
    R = max(0, min(R, img.width))
    T = max(0, min(T, img.height))
    B = max(0, min(B, img.height))
    return (L, T, R, B)


def spoint(img: Image.Image, x: int, y: int) -> tuple[int, int]:
    sx, sy = _scale(img)
    X = int(round(x * sx))
    Y = int(round(y * sy))
    # Apply detected UI offset
    X += UI_OFFSET_PIXELS[0]
    Y += UI_OFFSET_PIXELS[1]
    # Clamp to image bounds
    X = max(0, min(X, img.width - 1))
    Y = max(0, min(Y, img.height - 1))
    return (X, Y)


if not ALERT_WEBHOOK_URLS:
    logger.warning(
        "No Discord webhook URLs configured. Set DISCORD_WEBHOOK_URLS env var or [Log_Alert] alert_urls in Tribelog bot.ini"
    )
else:
    logger.debug(f"Loaded {len(ALERT_WEBHOOK_URLS)} Discord webhook URL(s)")

# Log detection area configuration
REGION_ENV = os.getenv("LOG_AREA") or config.get("Regions", "log_area", fallback="")


def _parse_log_area_region() -> tuple[int, int, int, int]:
    # Allow specifying absolute pixels via env/ini: "left,top,right,bottom"
    if REGION_ENV:
        try:
            parts = [int(p.strip()) for p in REGION_ENV.split(",")]
            if len(parts) == 4 and parts[2] > parts[0] and parts[3] > parts[1]:
                logger.info(f"Using configured log area: {parts}")
                return tuple(parts)  # type: ignore
        except Exception as e:
            logger.warning(f"Invalid LOG_AREA format '{REGION_ENV}': {e}")
    # Default to a center region ~400x160 px
    # Default to a center region ~400x160 px within the current image size
    w, h = 400, 160
    # Fallback center based on a typical 1920x1080 area if image size is unknown here
    cx, cy = 960, 540
    left = max(0, cx - w // 2)
    top = max(0, cy - h // 2)
    right = left + w
    bottom = top + h
    logger.info(f"Using default center log area: {(left, top, right, bottom)}")
    return (left, top, right, bottom)


LOG_AREA_RECT = _parse_log_area_region()


def get_log_area_rect(img: Image.Image) -> tuple[int, int, int, int]:
    """Return log area rectangle in absolute image coordinates.
    - If user provided explicit region via ini/env, use it as-is (clamped).
    - Otherwise, use design coords scaled and offset-aware via srect().
    """
    try:
        if REGION_ENV:
            l, t, r, b = LOG_AREA_RECT
            # Clamp to image bounds
            l = max(0, min(l, img.width))
            r = max(0, min(r, img.width))
            t = max(0, min(t, img.height))
            b = max(0, min(b, img.height))
            return (l, t, r, b)
        else:
            return srect(img, 760, 192, 1160, 820)
    except Exception:
        return srect(img, 760, 192, 1160, 820)


def _find_ark_window():
    # Try exact and fuzzy matches for the Ark window title
    candidates = []
    exact_titles = ["ArkAscended", "ARK: Survival Ascended"]

    for title in exact_titles:
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            candidates.append((hwnd, title))

    if not candidates:
        # Fallback: enumerate windows and find any that contain Ark keywords
        def _enum_handler(h, _):
            try:
                t = win32gui.GetWindowText(h)
                if t and any(k in t.lower() for k in ["ark", "ascended"]):
                    candidates.append((h, t))
            except Exception:
                pass

        win32gui.EnumWindows(_enum_handler, None)

    if candidates:
        hwnd, title = candidates[0]
        try:
            rect = win32gui.GetWindowRect(hwnd)
            logger.debug(f"Found Ark window '{title}' at rect={rect}")
        except Exception:
            logger.debug(f"Found Ark window '{title}'")
        return hwnd
    return 0


def is_ark_running():
    hwnd = _find_ark_window()
    if hwnd == 0:
        logger.error("Ark window not found. Ensure the game is running and visible.")
        return False
    return True


def _pil_to_gray_np(img: Image.Image):
    # Convert PIL Image to grayscale numpy array for cv2
    try:
        import numpy as _np
        import cv2 as _cv2

        arr = _np.array(img.convert("RGB"))
        gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)
        return gray
    except Exception:
        return None


_ANCHOR_TEMPLATE = None
_LAST_ANCHOR_SCORE = 0.0
_SCROLL_TEMPLATE = None
_LAST_SCROLL_POS = None  # type: Optional[tuple[int,int,int,int]]


def _load_anchor_template():
    global _ANCHOR_TEMPLATE
    if not CV2_AVAILABLE:
        return None
    try:
        if _ANCHOR_TEMPLATE is None and ANCHOR_FILE and Path(ANCHOR_FILE).exists():
            _ANCHOR_TEMPLATE = cv2.imread(ANCHOR_FILE, cv2.IMREAD_GRAYSCALE)
            if _ANCHOR_TEMPLATE is None:
                logger.debug(f"Failed to load anchor template: {ANCHOR_FILE}")
        return _ANCHOR_TEMPLATE
    except Exception as e:
        logger.debug(f"_load_anchor_template error: {e}")
        return None


def _update_ui_offset(img: Image.Image) -> tuple[bool, float]:
    """Detect UI offset by matching the anchor template near its expected location.
    Returns (found, score). Updates UI_OFFSET_PIXELS if found.
    """
    global UI_OFFSET_PIXELS, _LAST_ANCHOR_SCORE
    try:
        if not CV2_AVAILABLE:
            return False, 0.0
        tpl = _load_anchor_template()
        if tpl is None:
            return False, 0.0
        gray = _pil_to_gray_np(img)
        if gray is None:
            return False, 0.0
        th, tw = tpl.shape[:2]
        sx, sy = _scale(img)
        exp_x = int(round(ANCHOR_BASE_X * sx))
        exp_y = int(round(ANCHOR_BASE_Y * sy))
        margin = int(round(ANCHOR_SEARCH_MARGIN * sx))
        x0 = max(0, exp_x - margin)
        y0 = max(0, exp_y - margin)
        x1 = min(img.width, exp_x + margin)
        y1 = min(img.height, exp_y + margin)
        # Ensure region is at least the size of template
        if (x1 - x0) < tw or (y1 - y0) < th:
            # Fallback: search whole image (slower)
            x0, y0, x1, y1 = 0, 0, img.width, img.height
        region = gray[y0:y1, x0:x1]
        if region.shape[0] < th or region.shape[1] < tw:
            return False, 0.0
        res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = float(max_val)
        if score >= ANCHOR_MIN_SCORE:
            match_x = x0 + max_loc[0]
            match_y = y0 + max_loc[1]
            # Compute offset relative to expected top-left
            UI_OFFSET_PIXELS = [match_x - exp_x, match_y - exp_y]
            _LAST_ANCHOR_SCORE = score
            # Debug overlay image
            try:
                dbg = img.copy()
                draw = ImageDraw.Draw(dbg)
                draw.rectangle(
                    (match_x, match_y, match_x + tw, match_y + th),
                    outline="magenta",
                    width=2,
                )
                draw.rectangle((x0, y0, x1, y1), outline="orange", width=1)
                save_debug_image(dbg, "anchor_match", keep_only_latest=True)
            except Exception:
                pass
            logger.debug(f"Anchor matched: score={score:.2f} offset={UI_OFFSET_PIXELS}")
            return True, score
        else:
            _LAST_ANCHOR_SCORE = score
            logger.debug(f"Anchor not matched (score={score:.2f} < {ANCHOR_MIN_SCORE})")
            return False, score
    except Exception as e:
        logger.debug(f"_update_ui_offset error: {e}")
        return False, 0.0


def take_screenshot():
    try:
        logger.info("Taking SS")
        hwnd = _find_ark_window()
        if hwnd == 0:
            logger.error("Ark window not found")
            return None

        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        # Save rolling screenshot for immediate inspection (in debug folder)
        _ensure_debug_dir()
        img.save(ROLLING_SCREENSHOT)

        # Try to update UI offset based on template matching (if available)
        try:
            found, score = _update_ui_offset(img)
            if found:
                logger.debug(
                    f"UI offset applied: {UI_OFFSET_PIXELS} (anchor score {score:.2f})"
                )
            else:
                logger.debug("UI offset not updated (no anchor match)")
        except Exception as _e:
            logger.debug(f"UI offset check failed: {_e}")

        # Not saving timestamped screenshots to avoid duplicates
        return img
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        return None


def check_logs_open(img):
    # logger.debug("Checking if logs view is open (top-left TRIBES)")
    coords = srect(img, 176, 115, 274, 155)
    tribes_area = img.crop(coords)
    # logger.debug(f"TRIBES crop: {coords} (scaled from 176,115,274,155)")
    # Enhance for OCR robustness
    ta = tribes_area.convert("L")
    ta = ImageOps.autocontrast(ta)
    ta = ta.filter(ImageFilter.MedianFilter())
    _ensure_debug_dir()
    ta.save(TRIBES_CHECK_PATH)
    # If no anchor template exists yet, bootstrap one from current TRIBES crop
    try:
        if ANCHOR_FILE and not Path(ANCHOR_FILE).exists():
            Path(ANCHOR_FILE).parent.mkdir(parents=True, exist_ok=True)
            ta.save(ANCHOR_FILE)
            logger.debug(f"Bootstrapped anchor template at {ANCHOR_FILE}")
    except Exception:
        pass

    try:
        custom_config = "--oem 3 --psm 7 -l eng"
        text = pytesseract.image_to_string(ta, config=custom_config)
        global LAST_TRIBES_OCR
        LAST_TRIBES_OCR = text
        logger.debug(f"TRIBES OCR RAW='{LAST_TRIBES_OCR.strip()}'")
        text_clean = text.replace("\n", "").replace(" ", "").upper()
        ok = "TRIBES" in text_clean or "TRIBE" in text_clean
        return ok
    except Exception as e:
        logger.error(f"TRIBES OCR error: {e}")
        return False


def _load_scroll_template():
    global _SCROLL_TEMPLATE
    if not CV2_AVAILABLE:
        return None
    try:
        if _SCROLL_TEMPLATE is None:
            cand = [
                str(TEMPLATES_DIR / "scroll_thumb.png"),
                str(TEMPLATES_DIR / "scroll_thumb_small.png"),
            ]
            for p in cand:
                if Path(p).exists():
                    _SCROLL_TEMPLATE = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    break
        return _SCROLL_TEMPLATE
    except Exception:
        return None


def is_scrolled_to_old_logs(img):
    # Check scroll slider at exact position 1176,216 to 1176,285
    # If light slider color is at these coords = at bottom (current logs)
    # If NOT at these coords = scrolled up (old logs)
    try:
        # Save the slider column for debug
        slider_strip = img.crop(srect(img, 1168, 210, 1184, 290))
        # Always keep only the latest slider strip for clarity
        save_debug_image(slider_strip, "scroll_slider_strip", keep_only_latest=True)

        # Prefer template matching if template available
        tpl = _load_scroll_template()
        if CV2_AVAILABLE and tpl is not None:
            gray = _pil_to_gray_np(img)
            if gray is not None:
                res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                th, tw = tpl.shape[:2]
                # Draw match rect for debug
                try:
                    dbg = img.copy()
                    draw = ImageDraw.Draw(dbg)
                    draw.rectangle(
                        (max_loc[0], max_loc[1], max_loc[0] + tw, max_loc[1] + th),
                        outline="yellow",
                        width=2,
                    )
                    save_debug_image(dbg, "scroll_match", keep_only_latest=True)
                except Exception:
                    pass
                # Heuristic: if y of match is near the bottom of log area, we're at bottom
                l, t, r, b = get_log_area_rect(img)
                match_center_y = max_loc[1] + th // 2
                # Bottom band of the log area
                bottom_band_top = b - int(round(0.15 * (b - t)))
                is_at_bottom = match_center_y >= bottom_band_top
                logger.debug(
                    f"Scroll match score={max_val:.2f}, y={match_center_y}, bottom_band_top={bottom_band_top}"
                )
                return not is_at_bottom

        slider_pixels_found = 0
        total_pixels_checked = 0

        slider_color = (183, 246, 253)  # B7F6FD - light slider color

        # Check the exact vertical range
        for y in range(216, 286):  # 216 to 285 inclusive
            try:
                px = spoint(img, 1176, y)
                pixel = img.getpixel(px)
                total_pixels_checked += 1

                # Check if pixel matches slider color (with tolerance)
                color_diff = sum(abs(a - b) for a, b in zip(pixel, slider_color))
                if color_diff < 30:  # Tighter tolerance
                    slider_pixels_found += 1

            except:
                continue

        # If we find slider color in this exact area, we're at bottom (current logs)
        # If not, we're scrolled up (old logs)
        is_at_bottom = slider_pixels_found >= 67  # Need 68+ pixels to be at bottom

        # Only report if slider has 68+ pixels
        if slider_pixels_found >= 68:
            logger.debug(f"Newest logs shown (at top)")
        else:
            logger.debug(
                f"Likely scrolled to older logs (not at top), {slider_pixels_found}"
            )

        return not is_at_bottom  # Return True if scrolled to old logs

    except Exception as e:
        logger.debug(f"Scroll detection error: {e}")
        return False


def create_log_report(img):
    # Create log report image for Discord
    try:
        # Get larger log area for Discord - extend 250px lower
        left, top, right, bottom = srect(img, 760, 190, 1160, 500)
        # Extend bottom for context
        bottom = min(img.height, bottom + int(round(250 * _scale(img)[1])))
        log_report_area = img.crop((left, top, right, bottom))

        # Save into debug_screens folder and return that path
        save_debug_image(log_report_area, "log_report", keep_only_latest=True)
        out_path = DEBUG_DIR / "log_report.png"
        return str(out_path)
    except Exception as e:
        logger.debug(f"create_log_report failed: {e}")
        return None


def enhance_for_ocr(img):
    """Enhance image for better OCR accuracy"""
    img = img.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Reduced from 2 to avoid over-enhancement
    return img


def debug_ocr_region(img):
    # Save the OCR region with overlay to help debug positioning
    try:
        # Create a copy of the full image
        debug_img = img.copy().convert("RGB")

        # Draw rectangle around the OCR region

        draw = ImageDraw.Draw(debug_img)

        # Log area rectangle
        log_rect = get_log_area_rect(img)
        draw.rectangle(log_rect, outline="red", width=3)

        # Header area rectangle (780, 215, 933, 236)
        header_rect = srect(img, 780, 215, 933, 236)
        draw.rectangle(header_rect, outline="blue", width=2)

        # TRIBES check area (176, 115, 274, 155)
        tribes_rect = srect(img, 176, 115, 274, 155)
        draw.rectangle(tribes_rect, outline="green", width=2)

        # Parasaur bar marker (triangle) at top bar region (0, 0, 1920, 40)
        parasaur_rect = srect(img, 0, 0, 1920, 40)
        draw.rectangle(parasaur_rect, outline="cyan", width=2)

        # Scroll slider area (right edge of log)
        slider_rect = srect(img, 1168, 210, 1184, 290)
        draw.rectangle(slider_rect, outline="yellow", width=2)

        # Map area used to OCR 'who'
        map_rect = srect(img, *MAP_AREA_BASE_RECT)
        draw.rectangle(map_rect, outline="white", width=2)
        
        # Anchor search region (for visibility)
        sx, sy = _scale(img)
        exp_x = int(round(ANCHOR_BASE_X * sx))
        exp_y = int(round(ANCHOR_BASE_Y * sy))
        margin = int(round(ANCHOR_SEARCH_MARGIN * sx))
        x0 = max(0, exp_x - margin)
        y0 = max(0, exp_y - margin)
        x1 = min(img.width, exp_x + margin)
        y1 = min(img.height, exp_y + margin)
        draw.rectangle((x0, y0, x1, y1), outline="magenta", width=1)
        # Mark expected anchor point and current offset
        draw.ellipse(
            (exp_x - 3, exp_y - 3, exp_x + 3, exp_y + 3), outline="white", width=2
        )
        draw.text(
            (exp_x + 6, exp_y + 6),
            f"offset={UI_OFFSET_PIXELS} score={_LAST_ANCHOR_SCORE:.2f}",
            fill="white",
        )

        # Save debug image
        save_debug_image(debug_img, "ocr_regions_overlay", keep_only_latest=True)

        # Also save just the log area for inspection
        log_area = img.crop(log_rect)
        save_debug_image(log_area, "current_log_area", keep_only_latest=True)

        logger.debug(
            "Saved OCR debug images: ocr_regions_overlay.png and current_log_area.png"
        )

    except Exception as e:
        logger.debug(f"Debug OCR region failed: {e}")


# Canonical map names for autocorrection
MAP_CANONICAL_NAMES = [
    "Extinction",
    "Ragnarok",
    "Astraeos",
    "Aberration",
    "The Island",
    "Scorched Earth",
    "The Center",
]

def _normalize_key(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()

def autocorrect_map_name(name: str) -> str:
    try:
        key = _normalize_key(name)
        if not key:
            return name.strip()
        best = None
        best_score = 0.0
        for cand in MAP_CANONICAL_NAMES:
            score = difflib.SequenceMatcher(None, key, _normalize_key(cand)).ratio()
            if score > best_score:
                best_score = score
                best = cand
        # Require a reasonable similarity to avoid wrong corrections
        if best and best_score >= 0.6:
            return best
        return name.strip()
    except Exception:
        return name.strip()


def ocr_who_from_map(img: Image.Image) -> str | None:
    """OCR the configured map area and assign the result to global 'who'.
    Saves the crop for debugging in debug_screens/who_map_area.png
    Returns the detected string or None.
    """
    global who, WHO_LOCKED
    try:
        # Scale-aware rect with UI offset
        l, t, r, b = srect(img, *MAP_AREA_BASE_RECT)
        crop = img.crop((l, t, r, b)).convert("L")
        # light preprocessing for OCR
        crop = ImageOps.autocontrast(crop)
        crop = crop.filter(ImageFilter.MedianFilter())
        save_debug_image(crop, "who_map_area", keep_only_latest=True)

        # favor a single line; allow letters, digits, spaces and a few separators
        cfg = (
            "--oem 3 --psm 7 -l eng "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -c preserve_interword_spaces=1"
        )
        raw = pytesseract.image_to_string(crop, config=cfg)
        text = raw.strip()
        # normalize whitespace
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        # Ensure word-number separation by a single space
        m = re.match(r"^(.*?)[\s_\-:]*([0-9]+)$", text)
        if m:
            left = m.group(1).strip()
            right = m.group(2)
            text = f"{left} {right}" if left else right
        else:
            text = re.sub(r"([A-Za-zÄÖÜäöüß])\s*([0-9])", r"\1 \2", text)
            text = re.sub(r"([0-9])\s*([A-Za-zÄÖÜäöüß])", r"\1 \2", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        # sanity trims
        if text:
            # avoid extremely long usernames
            text = text[:60]
            # try to autocorrect known map names at the start of the string
            parts = text.split(" ", 1)
            if parts:
                corrected = autocorrect_map_name(parts[0])
                if len(parts) > 1:
                    text = f"{corrected} {parts[1]}"
                else:
                    text = corrected

            who = text
            WHO_LOCKED = True
            logger.info(f"WHO detected from map area: '{who}'")
            return who
        else:
            logger.debug("WHO OCR returned empty text")
            return None
    except Exception as e:
        logger.debug(f"ocr_who_from_map error: {e}")
        return None


def _fix_common_ocr_mistakes(s: str) -> str:
    """Normalize frequent OCR mistakes in ARK log text without being too aggressive."""
    # Normalize various forms of Day
    s = re.sub(r"\b[DO0]ay\b", "Day", s, flags=re.IGNORECASE)
    # Fix level notation e.g., LvI400, Lvi398 -> Lvl 400 / Lvl 398
    s = re.sub(r"\bLv[Il1]\s*(\d+)", r"Lvl \1", s)
    # Collapse repeated punctuation and spaces
    s = re.sub(r"[\s\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    # Remove stray spaces before punctuation
    s = re.sub(r"\s+([,.:;()\[\]])", r"\1", s)
    return s.strip()


def _remove_all_ingame_timestamps(text: str) -> str:
    """Remove or replace all occurrences of in-game timestamps in a string."""
    pattern = r"[DO0]ay\s*\d+\s*[,.: ]+\s*\d{1,2}[:.]\d{2}[:.]\d{2}[:.]?"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def _parse_ingame_timestamp(ts: str) -> int:
    """Parse 'Day N, hh:mm[:ss]' or variants into a comparable integer seconds value.
    Accepts 2 or 3 time components and both ':' or '.' separators.
    Returns 0 on failure.
    """
    try:
        m = re.search(
            r"Day\s*(\d+)[,.: ]+\s*(\d{1,2})(?::|\.)(\d{2})(?::|\.)?(\d{2})?",
            ts,
            flags=re.IGNORECASE,
        )
        if not m:
            return 0
        day = int(m.group(1))
        hh = int(m.group(2))
        mm = int(m.group(3))
        ss = int(m.group(4)) if m.group(4) else 0
        return day * 86400 + hh * 3600 + mm * 60 + ss
    except Exception:
        return 0


def check_for_changes(img, viewing_old_logs):
    """
    Always processes logs when visible, using header OCR for change detection.
    """
    if viewing_old_logs:
        logger.debug("[check_for_changes] Skipping: viewing old logs.")
        return False, []

    # Save the current log area for debugging context
    left, top, right, bottom = srect(img, 760, 192, 1160, 820)
    log_area = img.crop((left, top, right, bottom))
    _ensure_debug_dir()
    log_area.save(LOG_NEW_PATH)
    save_debug_image(log_area, "log_new_area", keep_only_latest=True)

    # Read the header text which contains the 'Final entry: Day ...' message
    header_text = _read_log_header_text(img)
    if not header_text:
        logger.debug(
            "[check_for_changes] Header OCR empty; cannot decide by 'Final entry' message."
        )
        return False, []

    current_ts, fmt_time = _parse_ingame_timestamp(header_text)
    logger.info(f"[check_for_changes] Current header time: {fmt_time}")

    # Always process logs if header time is valid
    if current_ts > 0:
        log_lines = extract_log_lines(img)
        processed_lines = process_log_lines(log_lines)
        if processed_lines:
            logger.success(
                f"[check_for_changes] Found {len(processed_lines)} new log entries."
            )
            return True, processed_lines
        else:
            logger.info("[check_for_changes] No new log entries found.")
            return False, []
    else:
        logger.debug(
            f"[check_for_changes] Could not parse header time: '{header_text}'"
        )
        return False, []


def extract_log_lines(img):
    """
    Extract log lines and return (ingame_ts, text) pairs.
    Each log entry should be separate, not combined into one long sentence.
    """
    try:
        # Use the configured/scaled region for logs
        l, t, r, b = get_log_area_rect(img)
        full_log_area = img.crop((l, t, r, b)).convert("L")
        _ensure_debug_dir()
        full_log_area.save(FULL_LOGS_OCR_PATH)

        config_simple = "--oem 3 --psm 6 -l eng"
        raw_text = pytesseract.image_to_string(full_log_area, config=config_simple)
        raw_text = _fix_common_ocr_mistakes(raw_text)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        logger.debug(
            f"[extract_log_lines] OCR extracted {len(lines)} lines. Sample: {lines[0][:80] if lines else 'No lines found'}"
        )

        log_entries = []
        for idx, line in enumerate(lines):
            timestamp_match = re.match(
                r"^(Day\s*\d+[,.: ]+\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)",
                line,
                re.IGNORECASE,
            )
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                ingame_ts = _parse_ingame_timestamp(timestamp_str)
                if isinstance(ingame_ts, tuple):
                    ingame_ts = ingame_ts[0]
                if ingame_ts > 0:
                    log_entries.append((ingame_ts, line))
                    logger.debug(
                        f"[extract_log_lines] Entry {idx+1}: ts={ingame_ts}, text='{line[:60]}'"
                    )

        logger.info(f"[extract_log_lines] Returning {len(log_entries)} log entries.")
        return log_entries
    except Exception as e:
        logger.error(f"[extract_log_lines] Failed: {e}")
        return []


def process_log_lines(log_lines):
    """Replace in-game time with Discord timestamp and filter by increasing ingame_ts.
    Returns a list of messages, one per log line, to send separately.
    """
    if not log_lines:
        return []

    processed_lines = []
    current_time = int(time.time())

    last_ingame_ts = _get_last_ingame_ts()
    max_new_ingame_ts = last_ingame_ts

    idx = 0
    for entry in log_lines:
        try:
            if isinstance(entry, tuple):
                ingame_ts, line = entry
            else:
                ingame_ts, line = 0, entry

            # Only process lines with a valid and strictly newer ingame timestamp
            if not ingame_ts or ingame_ts <= last_ingame_ts:
                logger.debug(
                    f"Skipping entry with ts {ingame_ts} (last={last_ingame_ts})"
                )
                continue

            # Remove all in-game timestamps from the string, even if multiple
            line = _remove_all_ingame_timestamps(line)
            line = re.sub(r"^[:. ]+", "", line)  # Clean leftovers
            line = _fix_common_ocr_mistakes(line)
            line = line.strip()

            # Filter out very short or meaningless lines
            if len(line) < 10:
                logger.debug(f"Skipping short line: '{line}'")
                continue

            # Filter out lines that look like OCR garbage
            if not re.search(
                r"[a-zA-Z]{3,}", line
            ):  # Must have at least one 3+ letter word
                logger.debug(f"Skipping garbage line: '{line}'")
                continue

            ts = current_time + idx  # stagger timestamps slightly for ordering
            formatted_line = f"<t:{ts}:T> {line}"
            processed_lines.append(formatted_line)

            if ingame_ts and ingame_ts > max_new_ingame_ts:
                max_new_ingame_ts = ingame_ts
            idx += 1

        except Exception as e:
            logger.warning(f"Line processing failed: {e}")
            continue

    if max_new_ingame_ts > last_ingame_ts:
        _update_last_ingame_ts(max_new_ingame_ts)
        logger.debug(
            f"Updated last ingame timestamp from {last_ingame_ts} to {max_new_ingame_ts}"
        )

    logger.debug(
        f"Processed {len(processed_lines)} lines from {len(log_lines)} input entries"
    )
    return processed_lines


def _hash_image(img: Image.Image) -> str:
    try:
        h = hashlib.sha256()
        h.update(img.tobytes())
        return h.hexdigest()
    except Exception:
        return ""


def _canonicalize_header_text(text: str) -> str:
    """Canonicalize header to 'DAY <day>, HH:MM[:SS]' if possible.
    Falls back to a safe normalized uppercase form if parsing fails.
    """
    raw = text
    text = text.strip().upper()
    # Fix common misreads of DAY
    text = re.sub(r"^(DA[VY]|DAV|DAYY|D4Y)", "DAY", text)
    # Common digit/letter swaps
    text = text.replace("O", "0").replace("I", "1").replace("L", "1")

    # Try to parse a tolerant pattern from the raw text, allowing missing chars/spaces
    try:
        m = re.search(
            r"D[AO0]Y\s*(\d+)\s*[,.: ]+\s*(\d{1,2})\s*[:\.]\s*(\d{2})(?:\s*[:\.]\s*(\d{2}))?",
            raw,
            flags=re.IGNORECASE,
        )
        if not m:
            m = re.search(
                r"DAY\s*(\d+)\s*[,.: ]+\s*(\d{1,2})\s*[:\.]\s*(\d{2})(?:\s*[:\.]\s*(\d{2}))?",
                text,
                flags=re.IGNORECASE,
            )
        if m:
            day = int(m.group(1))
            hh = int(m.group(2))
            mm = int(m.group(3))
            ss = int(m.group(4)) if m.group(4) else None
            if 0 <= hh <= 99 and 0 <= mm < 60 and (ss is None or 0 <= ss < 60):
                if ss is None:
                    return f"DAY {day}, {hh:02d}:{mm:02d}"
                return f"DAY {day}, {hh:02d}:{mm:02d}:{ss:02d}"
    except Exception:
        pass

    # Fallback: keep only allowed characters and collapse repeats
    text = re.sub(r"[^A-Z0-9:., ]", "", text)
    text = re.sub(r"([:., ])\1+", r"\1", text)
    return text.strip()


def _headers_equivalent(h1: str, h2: str) -> bool:
    """Return True if headers are effectively the same, allowing small OCR digit differences.
    Treat headers that only differ in seconds as equivalent to reduce noise.
    """
    if not h1 and not h2:
        return True
    if not h1 or not h2:
        return False

    # Canonicalize
    c1 = _canonicalize_header_text(h1)
    c2 = _canonicalize_header_text(h2)
    if c1 == c2:
        return True

    # Compare day and HH:MM only
    m1 = re.search(r"DAY\s*(\d+)\s*,\s*(\d{2}):(\d{2})", c1)
    m2 = re.search(r"DAY\s*(\d+)\s*,\s*(\d{2}):(\d{2})", c2)
    if m1 and m2:
        try:
            day1, hh1, mm1 = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
            day2, hh2, mm2 = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            if day1 == day2 and hh1 == hh2 and mm1 == mm2:
                return True
            # If day number is off by 1 due to OCR error but time matches closely, consider equal
            if abs(day1 - day2) <= 1 and hh1 == hh2 and mm1 == mm2:
                return True
        except ValueError:
            pass
    return False


def _read_log_header_text(img: Image.Image) -> str:
    """
    Crop and OCR the header region (Day/time), canonical form.
    """
    header = img.crop(srect(img, 780, 215, 933, 236)).convert("L")
    # If log area was explicitly overridden, try to align header slightly above it
    try:
        if REGION_ENV:
            l, t, r, b = LOG_AREA_RECT
            # The header is typically near the top of the log area; adjust with padding
            pad = int(round(25 * _scale(img)[1]))
            header = img.crop((l + 20, max(0, t - pad), r - 20, t + pad)).convert("L")
    except Exception:
        pass
    header = ImageOps.autocontrast(header)
    header = header.filter(ImageFilter.MedianFilter())
    header.save(HEADER_PATH)
    try:
        cfg = "--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:,."
        raw_text = pytesseract.image_to_string(header, config=cfg)
        canon = _canonicalize_header_text(raw_text)
        return canon
    except Exception as e:
        logger.debug(f"Header OCR error: {e}")
        return ""


def check_for_changes(img, viewing_old_logs):
    """Decide newness ONLY by the 'Final entry: Day N, HH:MM[:SS]' header in the tribe log.
    We OCR the header area and compare its parsed in-game timestamp against the last stored one.
    """

    # Optional cooldown to reduce spam
    if _is_on_cooldown():
        logger.debug("Skipping change check - still on cooldown")
        return False, []

    # If scrolled to old logs, avoid evaluating to prevent false positives
    if viewing_old_logs:
        logger.debug("Scrolled to old logs, ignoring changes")
        return False, []

    # Save the current log area for debugging context
    left, top, right, bottom = srect(img, 760, 192, 1160, 820)
    log_area = img.crop((left, top, right, bottom))
    _ensure_debug_dir()
    log_area.save(LOG_NEW_PATH)
    save_debug_image(log_area, "log_new_area", keep_only_latest=True)

    # Read the header text which contains the 'Final entry: Day ...' message
    header_text = _read_log_header_text(img)
    if not header_text:
        logger.debug("Header OCR empty; cannot decide by 'Final entry' message")
        return False, []

    # Parse Day/time from header. Try the standard parser first, then fallbacks tolerant to OCR quirks.
    final_ingame_ts = _parse_ingame_timestamp(header_text)

    if final_ingame_ts <= 0:
        # Fallback A: tolerant spaced/separator pattern
        m = re.search(
            r"D[AO0]Y\s*(\d+)\s*[,.: ]+\s*(\d{1,2})\s*[:\.]\s*(\d{2})(?:\s*[:\.]\s*(\d{2}))?",
            header_text,
            flags=re.IGNORECASE,
        )
        if m:
            try:
                day = int(m.group(1))
                hh = int(m.group(2))
                mm = int(m.group(3))
                ss = int(m.group(4)) if m.group(4) else 0
                final_ingame_ts = day * 86400 + hh * 3600 + mm * 60 + ss
            except Exception:
                final_ingame_ts = 0

    if final_ingame_ts <= 0:
        # Fallback B: digits-only OCR like 'DAY3071704814'
        try:
            digits = re.sub(r"\D+", "", header_text)
            # Remove any leading zeros to avoid hour parsing issues, but keep at least 6 for time
            digits = digits.lstrip("0") or digits
            hh = mm = ss = 0
            if len(digits) >= 7:  # assume last 6 are HHMMSS
                day = int(digits[:-6])
                tail = digits[-6:]
                hh = int(tail[:2])
                mm = int(tail[2:4])
                ss = int(tail[4:6])
            elif len(digits) >= 5:  # assume last 4 are HHMM
                day = int(digits[:-4])
                tail = digits[-4:]
                hh = int(tail[:2])
                mm = int(tail[2:4])
            else:
                day = 0
            if day > 0 and 0 <= hh <= 99 and 0 <= mm < 60 and 0 <= ss < 60:
                final_ingame_ts = day * 86400 + hh * 3600 + mm * 60 + ss
        except Exception:
            final_ingame_ts = 0

    if final_ingame_ts <= 0:
        logger.debug(f"Could not parse 'Final entry' header: '{header_text}'")
        return False, []

    last_ingame_ts = _get_last_ingame_ts()
    logger.debug((f"New timestamp: {final_ingame_ts} old timestamp: {last_ingame_ts}"))

    # If stored ts is ahead of header (possible stale/corrupt state), resync silently
    if last_ingame_ts and last_ingame_ts > final_ingame_ts:
        logger.warning(
            f"Stored last_ingame_ts ({last_ingame_ts}) is ahead of header ({final_ingame_ts}); resyncing to header"
        )
        _update_last_ingame_ts(final_ingame_ts)
        return False, []

    if final_ingame_ts > last_ingame_ts:
        _update_last_ingame_ts(final_ingame_ts)
        _update_last_change_time()
        logger.info(f"New Final entry detected: header advanced to {final_ingame_ts}")
        return True, []

    # Not newer; no change
    return False, []


def check_parasaur(img):
    # Parasaur area: top 30y coords
    parasaur_area = img.crop(srect(img, 400, 0, 756, 30))
    save_debug_image(parasaur_area, "parasaur_bar", keep_only_latest=True)
    pixels = list(parasaur_area.getdata())
    return (0, 255, 234) in pixels  # Cyan color


# --- Red surge detection ---
SURGE_COOLDOWN_SECONDS = 10  # avoid spamming
SURGE_RED_THRESHOLD = 0.05  # 5% of pixels in log area appear red-ish

# --- Periodic status updates ---
STATUS_UPDATE_INTERVAL_SECONDS = 600  # 10 minutes


def _last_surge_time() -> float:
    state = _load_state()
    return float(state.get("last_surge_time", 0.0) or 0.0)


def _update_last_surge_time(ts: float) -> None:
    state = _load_state()
    state["last_surge_time"] = float(ts)
    _save_state(state)


def _estimate_red_ratio(img: Image.Image) -> float:
    """Estimate ratio of red-ish pixels in log area.
    Uses a simple heuristic: R is significantly higher than G and B.
    """
    try:
        log_rect = (760, 192, 1160, 820)
        region = img.crop(srect(img, 760, 192, 1160, 820)).convert("RGB")
        w, h = region.size
        pixels = region.getdata()
        total = w * h
        red_count = 0
        # Heuristic thresholds; tweak as needed
        for r, g, b in pixels:
            if r > 120 and r > g + 30 and r > b + 30:
                red_count += 1
        return red_count / max(total, 1)
    except Exception:
        return 0.0


def debug_position(img, x1, y1, x2, y2, name="debug"):
    """Take screenshot of specific coordinates for debugging"""
    try:
        # Use exact coordinates
        debug_area = img.crop((x1, y1, x2, y2))

        # Save with descriptive name
        filename = f"debug_{name}_{x1}_{y1}_{x2}_{y2}.png"
        debug_area.save(filename)

        logger.info(f"Debug screenshot saved: {filename}")
        logger.info(f"Coordinates: {x1},{y1} to {x2},{y2}")

        return filename
    except Exception as e:
        logger.error(f"Debug position error: {e}")
        return None


def _chunk_text(text: str, limit: int = 1800) -> List[str]:
    # Keep some headroom for mentions/formatting; Discord hard limit is 2000
    chunks: List[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks or [""]


def _send_webhook_once(url: str, content: str, image_path: Optional[str], username_override: Optional[str] = None) -> bool:
    try:
        webhook = DiscordWebhook(
            url=url,
            content=content,
            username=username_override if username_override else who,
            rate_limit_retry=True,  # library will honor Retry-After
            timeout=15,
        )
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                webhook.add_file(file=f.read(), filename=os.path.basename(image_path))
        response = webhook.execute()
        # The library may return Response or list[Response]
        if isinstance(response, list) and response:
            response = response[0]
        if hasattr(response, "status_code"):
            sc = response.status_code
            if 200 <= sc < 300:
                logger.debug(f"Webhook ok ({sc}) -> {url[:32]}...")
                return True
            else:
                body = getattr(response, "text", "")
                logger.error(f"Webhook error {sc}: {body}")
                return False
        else:
            logger.warning("No response object returned by webhook library")
            return False
    except Exception as e:
        logger.error(f"Webhook exception: {e}")
        return False


def send_alert(message: str, image_path: Optional[str] = None, username_override: Optional[str] = None) -> bool:
    if not ALERT_WEBHOOK_URLS:
        logger.error("send_alert called but no webhook URLs configured")
        return False

    # Ensure roles mention is preserved
    content = message

    # Respect Discord 2000-char limit with small safety margin
    all_chunks = _chunk_text(content, limit=1800)

    success_any = False
    for idx, chunk in enumerate(all_chunks):
        sent_this_chunk = False
        # Only attach image on the first chunk to avoid duplicates
        attach_image = image_path if idx == 0 else None
        # Try all configured webhooks until one succeeds for this chunk
        for url in ALERT_WEBHOOK_URLS:
            # Quick retry loop per URL
            attempts = 0
            delay = 2
            while attempts < 3 and not sent_this_chunk:
                attempts += 1
                if _send_webhook_once(url, chunk, attach_image, username_override=username_override):
                    sent_this_chunk = True
                    success_any = True
                    break
                time.sleep(delay)
                delay = min(delay * 2, 15)
        if not sent_this_chunk:
            logger.error("Failed to deliver a message chunk to all webhooks")
    return success_any


def main_loop():
    logger.info(f"Starting {who}")
    failed_attempts = 0
    quick_retries = 0
    last_parasaur_alert = 0  # Track parasaur alert cooldown
    last_status_update = 0  # Track 10-minute status update cadence
    status_update_count = 0  # Number of status updates sent successfully
    debug_counter = 0  # For periodic debug saves

    while True:
        try:
            if not is_ark_running():
                logger.debug("ARK not running, waiting...")
                time.sleep(10)
                continue

            img = take_screenshot()
            if not img:
                logger.debug("Screenshot failed, retrying...")
                time.sleep(5)
                continue

            # Detect 'who' dynamically from the map area once per run
            if not WHO_LOCKED:
                ocr_who_from_map(img)

            # Save debug OCR regions every 10 cycles for troubleshooting
            debug_counter += 1
            if debug_counter % 10 == 1:
                debug_ocr_region(img)

            if not check_logs_open(img):
                logger.debug("Logs not open, retrying in 3 seconds")
                time.sleep(3)
                continue

            # Reset counters when logs are open
            failed_attempts = 0
            quick_retries = 0

            # Check if we're viewing old logs
            viewing_old_logs = is_scrolled_to_old_logs(img)

            # Warm start: initialize last_ingame_ts to current header on first run
            state = _load_state()
            if not state.get("_warm_started", False):
                header_now = _read_log_header_text(img)
                # Extract comparable timestamp from header using tolerant parsing
                warm_ts = _parse_ingame_timestamp(header_now)
                if warm_ts <= 0:
                    try:
                        digits = re.sub(r"\D+", "", header_now)
                        digits = digits.lstrip("0") or digits
                        hh = mm = ss = 0
                        if len(digits) >= 7:
                            day = int(digits[:-6])
                            tail = digits[-6:]
                            hh = int(tail[:2])
                            mm = int(tail[2:4])
                            ss = int(tail[4:6])
                        elif len(digits) >= 5:
                            day = int(digits[:-4])
                            tail = digits[-4:]
                            hh = int(tail[:2])
                            mm = int(tail[2:4])
                            ss = 0
                        else:
                            day = 0
                        if day > 0 and 0 <= hh <= 99 and 0 <= mm < 60 and 0 <= ss < 60:
                            warm_ts = day * 86400 + hh * 3600 + mm * 60 + ss
                    except Exception:
                        warm_ts = 0
                if warm_ts > 0:
                    _update_last_ingame_ts(warm_ts)
                    state["_warm_started"] = True
                    _save_state(state)
                    logger.info(
                        f"Warm start: set last_ingame_ts to header time {header_now} -> {warm_ts}"
                    )
                else:
                    # Even if header missing, mark warm started to avoid repeating
                    state["_warm_started"] = True
                    _save_state(state)
                    logger.info(
                        "Warm start: header parsing failed; marked warm started to avoid flood"
                    )

            # Check for parasaur (only when viewing current logs and not on cooldown)
            current_time = time.time()
            if (current_time - last_parasaur_alert) > 30:  # 30 second cooldown
                if check_parasaur(img):
                    logger.warning("Alert Found: Parasaur Ping Detected")
                    if send_alert(f"{ROLE_MENTION} Parasaur, Simply Too Close - {who}"):
                        last_parasaur_alert = current_time
                    else:
                        logger.error("Failed to send parasaur alert")

            # Red surge alert (only when viewing current logs)
            if not viewing_old_logs:
                red_ratio = _estimate_red_ratio(img)
                logger.debug(f"Red ratio in log area: {red_ratio:.3f}")
                if red_ratio >= SURGE_RED_THRESHOLD:
                    last_surge = _last_surge_time()
                    if (current_time - last_surge) >= SURGE_COOLDOWN_SECONDS:
                        msg = f"{ROLE_MENTION} RAID DETECTED DO SMTH OR JAMES TAMES R GONE ({red_ratio:.0%})"
                        logger.warning(msg)
                        send_alert(msg)
                        _update_last_surge_time(current_time)

            # Check for log changes (only when viewing current logs)
            if not viewing_old_logs:
                has_changes, _ = check_for_changes(img, viewing_old_logs)
                if has_changes:
                    # Send a single screenshot of the tribelog instead of parsed text lines
                    image_path = create_log_report(img)
                    # Send only the current unix time and Discord timestamp format
                    unix_ts = int(time.time())
                    msg = f"<t:{unix_ts}:T>"
                    if send_alert(msg, image_path=image_path):
                        logger.info("✓ Sent tribelog screenshot update")
                    else:
                        logger.error("âœ— Failed to send tribelog screenshot update")
                else:
                    logger.info("No new changes detected")

            else:
                logger.debug("Viewing old logs - skipping change detection")

            # Periodic 10-minute status update (global)
            if (time.time() - last_status_update) >= STATUS_UPDATE_INTERVAL_SECONDS:
                status_name = f"{who} - Status Update"
                status_msg = f"<t:{int(time.time())}:T>"
                # Always include image starting from the first update
                image_path = create_log_report(img)
                if send_alert(status_msg, image_path=image_path, username_override=status_name):
                    last_status_update = time.time()
                    status_update_count += 1
                    logger.info("✓ Sent periodic status update")
                else:
                    logger.error("Failed to send periodic status update")

            # Dynamic sleep based on activity
            if viewing_old_logs:
                time.sleep(5)  # Longer sleep when viewing old logs
            else:
                time.sleep(3)  # Normal monitoring interval

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            failed_attempts += 1
            logger.error(f"Main loop error (attempt {failed_attempts}): {e}")

            # Progressive backoff on repeated failures
            if failed_attempts <= 3:
                time.sleep(5)
            elif failed_attempts <= 10:
                time.sleep(15)
            else:
                logger.error("Too many failures, entering extended retry mode")
                time.sleep(60)
                failed_attempts = 0  # Reset after long sleep


if __name__ == "__main__":
    main_loop()
