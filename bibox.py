#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pymupdf", "cryptography"]
# ///
"""
Decrypt BiBox 2.0 offline-synced files and create searchable PDFs.

Reads the Chrome IndexedDB blob to extract the page-to-hash mapping,
decrypts each page image (AES-256-CTR), embeds invisible text overlay
from BiBox pageData (with full Unicode support), and combines them
into a searchable PDF.

Usage: bibox [--output <dir>] [--no-text] [--debug-text]
             [--save-images] [--no-materials] [--book <id>]
             [--markdown] [--force] [--half-res]
"""

import sys
import os
import re
import json
import struct
import subprocess
import tempfile
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# -- PyMuPDF for PDF creation + text overlay --
import fitz  # PyMuPDF


# -- BiBox AES-256-CTR constants (hardcoded in the Electron app) --
BIBOX_KEY = b"helloWorldhelloWorldhelloWorld32"
BIBOX_IV = bytes.fromhex("1234567890ab1234567890ab00000000")


def decrypt(buf: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(BIBOX_KEY), modes.CTR(BIBOX_IV))
    decryptor = cipher.decryptor()
    return decryptor.update(buf) + decryptor.finalize()


# -- Path helpers --
def hash_to_file_path(sync_dir: Path, h: str) -> Path:
    return sync_dir / h[:3] / h[3:6] / h[6:9] / h


# -- Varint decoding --
def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 128:
            break
    return result, pos


def decode_varint_zigzag(buf: bytes, pos: int) -> int:
    result = 0
    shift = 0
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 128:
            break
    return (result >> 1) ^ -(result & 1)


# -- Extract page mapping from LDB hashesA array (primary) --
def _extract_page_mapping_ldb(ldb_dir: Path, book_id: int) -> list[dict]:
    # Zigzag-encode book_id to varint bytes for marker search
    zz = book_id * 2
    varint = bytearray()
    while zz > 0x7F:
        varint.append((zz & 0x7F) | 0x80)
        zz >>= 7
    varint.append(zz)
    book_marker = b"bookIdI" + bytes(varint)

    md5_re = re.compile(rb"[0-9a-f]{32}")
    best: list[str] = []

    for ldb_file in sorted(ldb_dir.glob("*.ldb")) + sorted(ldb_dir.glob("*.log")):
        try:
            data = ldb_file.read_bytes()
        except Exception:
            continue
        idx = 0
        while True:
            pos = data.find(b"hashesA", idx)
            if pos == -1:
                break
            idx = pos + 1
            # Verify correct book_id appears within 300 bytes before
            if book_marker not in data[max(0, pos - 300):pos]:
                continue
            # Find first MD5 hash within 40 bytes after marker (4-byte preamble + 32-byte hash)
            after = pos + len("hashesA")
            m = md5_re.search(data, after, after + 40)
            if not m:
                continue
            # Greedily collect consecutive hashes (allow ≤10-byte gaps to bridge WAL block headers)
            hashes: list[str] = []
            cur = m.start()
            while cur + 32 <= len(data):
                chunk = data[cur:cur + 32]
                if md5_re.fullmatch(chunk):
                    hashes.append(chunk.decode("ascii"))
                    cur += 32
                else:
                    skipped = False
                    for skip in range(1, 11):
                        nxt = data[cur + skip:cur + skip + 32]
                        if len(nxt) == 32 and md5_re.fullmatch(nxt):
                            hashes.append(nxt.decode("ascii"))
                            cur += skip + 32
                            skipped = True
                            break
                    if not skipped:
                        break
            if len(hashes) > len(best):
                best = hashes

    return [{"page": i + 1, "hash": h} for i, h in enumerate(best)]


# -- Extract page mapping from Chrome IndexedDB blob (fallback) --
def _extract_page_mapping_blob(blob_path: Path) -> list[dict]:
    buf = blob_path.read_bytes()
    text = buf.decode("latin-1")

    url_re = re.compile(
        r"https://static\.bibox2\.westermann\.de/bookpages/[A-Za-z0-9+/=]+/(\d+)\.png"
    )
    # Fallback for fragmented entries where only the URL tail is readable
    tail_re = re.compile(r"/(\d{1,4})\.png")
    md5_re = re.compile(r"[0-9a-f]{32}")

    hashes = [(m.start(), m.group(0)) for m in md5_re.finditer(text)]

    def find_nearest_hash(u_end: int) -> str | None:
        for h_pos, h_val in hashes:
            if u_end < h_pos < u_end + 400:
                return h_val
        return None

    pairs = []
    url_ends: set[int] = set()

    for m in url_re.finditer(text):
        h = find_nearest_hash(m.end())
        if h:
            pairs.append({"page": int(m.group(1)), "hash": h, "pos": m.start()})
            url_ends.add(m.end())

    for m in tail_re.finditer(text):
        if m.end() in url_ends:
            continue
        h = find_nearest_hash(m.end())
        if h:
            pairs.append({"page": int(m.group(1)), "hash": h, "pos": m.start()})

    by_page: dict[int, list] = {}
    for p in pairs:
        by_page.setdefault(p["page"], []).append(p)

    if not by_page:
        return []

    pages = []
    for page in sorted(by_page.keys()):
        entries = sorted(by_page[page], key=lambda e: e["pos"])
        h = entries[1]["hash"] if len(entries) >= 2 else entries[0]["hash"]
        pages.append({"page": page, "hash": h})

    return pages


def _pixel_dims(data: bytes, ext: str) -> tuple[int, int]:
    """Return (width, height) in pixels from image header."""
    if ext == "png" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if ext == "jpg":
        i = 2
        while i + 4 <= len(data):
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            if data[i] == 0xFF and data[i + 1] in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + seg_len
    return 0, 0


def _image_dpi(data: bytes, ext: str) -> float:
    """Return DPI from PNG pHYs chunk or JPEG JFIF APP0, 0 if not found."""
    if ext == "png":
        pos = 8
        while pos + 12 <= len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            chunk = data[pos + 4:pos + 8]
            if chunk == b"pHYs" and pos + 8 + length <= len(data):
                px = struct.unpack(">I", data[pos + 8:pos + 12])[0]
                unit = data[pos + 16]
                return (px / 39.3701) if unit == 1 and px > 0 else 0.0
            if chunk == b"IDAT":
                break
            pos += 12 + length
    elif ext == "jpg":
        i = 2
        while i + 4 <= len(data):
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            if data[i:i + 2] == b"\xff\xe0" and data[i + 4:i + 9] == b"JFIF\x00":
                units = data[i + 11]
                xdpi = struct.unpack(">H", data[i + 12:i + 14])[0]
                if units == 1 and xdpi > 0:
                    return float(xdpi)
                if units == 2 and xdpi > 0:
                    return xdpi * 2.54
                break
            i += 2 + seg_len
    return 0.0


def extract_page_mapping(blob_path: Path, ldb_dir: Path | None = None, book_id: int | None = None) -> list[dict]:
    if ldb_dir and book_id:
        pages = _extract_page_mapping_ldb(ldb_dir, book_id)
        if pages:
            return pages
    return _extract_page_mapping_blob(blob_path)


# -- Extract book titles from LevelDB --
TITLE_MARKER = b'\x22\x05\x74\x69\x74\x6c\x65\x22'  # "\x05title"
ID_MARKER = b'\x22\x02\x69\x64\x49'                   # "\x02idI" (old format)
BOOK_ID_MARKER = b'bookIdI'                            # new format


def extract_book_titles(ldb_dir: Path, books_dir: Path | None = None) -> dict[int, str]:
    import json
    titles = {}

    # Seed from titles.json cache — survives both LDB compaction and folder deletion
    titles_cache = (books_dir / "titles.json") if books_dir else None
    if titles_cache and titles_cache.exists():
        try:
            for k, v in json.loads(titles_cache.read_text(encoding="utf-8")).items():
                titles[int(k)] = v
        except Exception:
            pass

    # Seed from existing output folder names — survives LDB compaction
    if books_dir and books_dir.exists():
        for d in books_dir.iterdir():
            if not d.is_dir():
                continue
            m = re.match(r"^(.+?)\s+\((\d+)\)$", d.name)
            if m:
                t, bid = m.group(1).strip(), int(m.group(2))
                if bid > 0 and len(t) > 2 and len(t) > len(titles.get(bid, "")):
                    titles[bid] = t

    for f in ldb_dir.iterdir():
        if f.suffix not in (".ldb", ".log"):
            continue
        buf = f.read_bytes()

        idx = 0
        while True:
            idx = buf.find(TITLE_MARKER, idx)
            if idx == -1:
                break

            title_start = idx + len(TITLE_MARKER)
            title_len, str_start = read_varint(buf, title_start)
            if title_len <= 0 or title_len > 300 or str_start + title_len > len(buf):
                idx += 8
                continue

            # Decode as Latin-1; title may have binary junk mid-way, stop at first control char
            raw = buf[str_start: str_start + title_len]
            title = re.split(rb'[\x00-\x1f]', raw)[0].decode("latin-1").strip()

            after = buf[str_start + title_len: str_start + title_len + 200]
            before = buf[max(0, idx - 200): idx]

            book_id = None
            # Old format: pagenumI after title + "\x02idI" before
            if b"pagenumI" in after:
                id_pos = before.rfind(ID_MARKER)
                if id_pos != -1:
                    book_id = decode_varint_zigzag(before, id_pos + len(ID_MARKER))
            # New format: bookIdI may straddle the title blob boundary — search overlapping window
            if not book_id:
                overlap_start = max(0, str_start + title_len - len(BOOK_ID_MARKER))
                overlap = buf[overlap_start: overlap_start + len(BOOK_ID_MARKER) + 200]
                bid_pos = overlap.find(BOOK_ID_MARKER)
                if bid_pos != -1:
                    book_id = decode_varint_zigzag(buf, overlap_start + bid_pos + len(BOOK_ID_MARKER))

            if book_id and book_id > 0 and len(title) > 1:
                if len(title) > len(titles.get(book_id, "")):
                    titles[book_id] = title

            idx = str_start + title_len

    return titles


# -- Extract book ID from blob --
def extract_book_id(blob_path: Path) -> int | None:
    buf = blob_path.read_bytes()
    pos = buf.find(b"bookIdI")
    if pos == -1:
        return None
    return decode_varint_zigzag(buf, pos + 7)


# -- Find all blob files with book data --
def find_blob_files(idb_blob_dir: Path) -> list[Path]:
    blobs = []
    if not idb_blob_dir.exists():
        return blobs

    for f in idb_blob_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            buf = f.read_bytes()
            text = buf.decode("latin-1")
            if "bookIdI" in text and "static.bibox2.westermann.de" in text:
                blobs.append(f)
        except Exception:
            pass

    return blobs


# -- Find all pageData JSONs --
def find_all_page_data(sync_dir: Path, ldb_dir: Path) -> list[dict]:
    hash_re = re.compile(rb"pageDataHash.{1,5}([0-9a-f]{32})")
    candidates = set()

    for f in ldb_dir.iterdir():
        if f.suffix not in (".ldb", ".log"):
            continue
        buf = f.read_bytes()
        for m in hash_re.finditer(buf):
            candidates.add(m.group(1).decode("ascii"))

    # Newer BiBox versions reference pageData via |hash= format in blobs rather than
    # pageDataHash in LDB. Scan sync_dir directly for JSON files as fallback.
    _ks0 = decrypt(b"\x00" * 16)[0]  # keystream byte 0 (same for every file)
    json_enc = _ks0 ^ 0x7B           # encrypted '{' character
    for f in sync_dir.rglob("*"):
        if not f.is_file() or f.name in candidates:
            continue
        try:
            with open(f, "rb") as fp:
                first = fp.read(1)
            if first and first[0] == json_enc:
                candidates.add(f.name)
        except Exception:
            pass

    results = []
    for h in candidates:
        file_path = hash_to_file_path(sync_dir, h)
        if not file_path.exists():
            continue

        try:
            dec = decrypt(file_path.read_bytes())
            text = dec.decode("utf-8")
            if not text.startswith("{"):
                continue
            data = json.loads(text)
            keys = list(data.keys())
            if keys and data[keys[0]] and "txt" in data[keys[0]]:
                results.append(data)
        except Exception:
            pass

    return results


# -- Extract supplemental material references from blob --
def extract_materials(blob_buf: bytes) -> list[dict]:
    title_key    = b'\x22\x05title\x22'     # "\x05title"
    file_key     = b'\x22\x04file\x22'       # "\x04file"
    filetype_key = b'\x22\x08filetype\x22'  # "\x08filetype"  — new format per-item anchor
    md5sum_key   = b'\x22\x06md5sum\x22\x20'  # "\x06md5sum" + space  — new format value prefix

    # Old format
    if b"materialsA" in blob_buf:
        md5_key = b'\x63\x0c\x6d\x00\x64\x00\x35\x00\x73\x00\x75\x00\x6d\x00'

        def read_len_str(pos: int) -> tuple[str, int]:
            length = 0
            shift = 0
            while pos < len(blob_buf):
                byte = blob_buf[pos]
                pos += 1
                length |= (byte & 0x7F) << shift
                shift += 7
                if byte < 128:
                    break
            s = blob_buf[pos: pos + length].decode("latin-1")
            return s, pos + length

        materials = []
        pos = blob_buf.find(b"materialsA")
        while True:
            pos = blob_buf.find(title_key, pos)
            if pos == -1:
                break
            t_str, t_end = read_len_str(pos + len(title_key))
            f_pos = blob_buf.find(file_key, t_end)
            if f_pos == -1 or f_pos > t_end + 200:
                pos = t_end
                continue
            f_str, f_end = read_len_str(f_pos + len(file_key))
            ext = f_str.rsplit(".", 1)[-1] if "." in f_str else ""
            md5sum = None
            m_pos = blob_buf.find(md5_key, f_end)
            if m_pos != -1 and m_pos < f_end + 4000:
                hash_start = m_pos + len(md5_key) + 2
                candidate = blob_buf[hash_start: hash_start + 32].decode("ascii", errors="replace")
                if re.fullmatch(r"[0-9a-f]{32}", candidate):
                    md5sum = candidate
            materials.append({"title": t_str, "file": f_str, "ext": ext, "md5sum": md5sum})
            pos = f_end
        return materials

    # New format: each item has "file" → "zipUrl" → "filetype" → "md5sum" sequence.
    # "preview_md5sum" uses key length 0x0E, so md5sum_key (0x06) never matches it.
    # Between "file" and "filetype" there may be large binary blobs (page images), so
    # we anchor backwards via "zipUrl" (always ≤100 bytes before "filetype") and then
    # search backwards from there for "file" and "title".
    zipurl_key = b'\x22\x06zipUrl'  # "\x06zipUrl" — no value-type suffix (varies)
    if filetype_key not in blob_buf:
        return []

    materials = []
    pos = 0
    while True:
        pos = blob_buf.find(filetype_key, pos)
        if pos == -1:
            break

        ft_len, ft_start = read_varint(blob_buf, pos + len(filetype_key))
        if not (0 < ft_len <= 10):
            pos += len(filetype_key)
            continue
        ext = blob_buf[ft_start: ft_start + ft_len].decode("latin-1", errors="replace")

        # Find "zipUrl" ≤100 bytes before "filetype" — it's the nearest anchor
        near_before = blob_buf[max(0, pos - 100): pos]
        zp = near_before.rfind(zipurl_key)
        if zp == -1:
            pos += len(filetype_key)
            continue
        # Absolute position of "zipUrl" in blob
        zipurl_abs = max(0, pos - 100) + zp

        # "file" and "title" appear ≤400 bytes before "zipUrl"
        before = blob_buf[max(0, zipurl_abs - 400): zipurl_abs]

        fp = before.rfind(file_key)
        filename = None
        if fp != -1:
            f_len, f_start = read_varint(before, fp + len(file_key))
            if 0 < f_len <= 150:
                filename = re.sub(rb'[\x00-\x1f\x7f]', b'', before[f_start: f_start + f_len]).decode("latin-1", errors="replace")

        if filename is None:
            # FILE_KEY missing or varint too large (binary blob separates fields):
            # scan backwards from zipUrl for a printable suffix (filename tail)
            fname_bytes = []
            for scan_pos in range(zipurl_abs - 1, max(0, zipurl_abs - 100), -1):
                b = blob_buf[scan_pos]
                if 0x20 <= b <= 0x7E:
                    fname_bytes.insert(0, b)
                else:
                    break
            filename = bytes(fname_bytes).decode("ascii", errors="replace").strip()

        tp = before.rfind(title_key)
        title = filename
        if tp != -1:
            t_len, t_start = read_varint(before, tp + len(title_key))
            if 0 < t_len <= 200:
                title = re.split(rb'[\x00-\x1f]', before[t_start: t_start + t_len])[0].decode("latin-1", errors="replace").strip()

        # "md5sum" field follows "filetype" within the same entry
        after = blob_buf[pos: pos + 600]
        md5sum = None
        mp = after.find(md5sum_key)
        if mp != -1:
            candidate = after[mp + len(md5sum_key): mp + len(md5sum_key) + 32]
            if re.fullmatch(rb'[0-9a-f]{32}', candidate):
                md5sum = candidate.decode("ascii")

        materials.append({"title": title, "file": filename, "ext": ext, "md5sum": md5sum})
        pos += len(filetype_key)

    return materials


# -- Extract words with bounding boxes from BiBox pageData --
def extract_words(txt: str, cds: list) -> list[dict]:
    if not txt or not cds:
        return []

    words = []

    def push_segment(text: str, start_idx: int, end_idx: int):
        fc = cds[start_idx] if start_idx < len(cds) else None
        lc = cds[end_idx] if end_idx < len(cds) else None
        if not fc or not lc or (fc[0] == 0 and fc[2] == 0):
            return
        w = (lc[1] - fc[0]) / 1000
        if w <= 0:
            return
        words.append({
            "text": text,
            "x": fc[0] / 1000,
            "w": w,
            "y": fc[2] / 1000,
            "h": (fc[3] - fc[2]) / 1000,
        })

    seg_start = -1
    seg_chars = ""

    for i in range(len(txt) + 1):
        ch = txt[i] if i < len(txt) else " "
        is_space = ch in " \n\t\r"

        if is_space:
            if seg_start != -1 and seg_chars:
                push_segment(seg_chars, seg_start, i - 1)
            seg_start = -1
            seg_chars = ""
            continue

        if seg_start == -1:
            seg_start = i
            seg_chars = ch
        else:
            prev_coord = cds[i - 1] if i - 1 < len(cds) else None
            cur_coord = cds[i] if i < len(cds) else None
            if (prev_coord and cur_coord and prev_coord[2] != 0 and cur_coord[2] != 0
                    and abs(cur_coord[2] - prev_coord[2]) > 500):
                if seg_chars:
                    push_segment(seg_chars, seg_start, i - 1)
                seg_start = i
                seg_chars = ch
            else:
                seg_chars += ch

    return words


# -- Clean up BiBox OCR text artifacts --
def clean_text(text: str) -> str:
    text = text.replace(" . ", ". ")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" ;", ";")
    text = text.replace(" :", ":")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    text = text.replace(". -", ".-")
    text = text.replace(":// ", "://")
    text = text.replace("www. ", "www.")
    text = re.sub(r"\. de\b", ".de", text)
    text = re.sub(r"\. com\b", ".com", text)
    text = re.sub(r"\. org\b", ".org", text)
    text = re.sub(r"\. net\b", ".net", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


# -- Format page text with structure detection --
def format_page_text(txt: str, cds: list, *, markdown: bool = False) -> str:
    if not txt or not cds:
        return ""

    # Build lines by y-coordinate
    lines = []
    line_chars = ""
    line_y = -1
    line_h = 0
    line_x = 99999
    char_count = 0
    height_sum = 0

    for i in range(len(txt)):
        c = cds[i] if i < len(cds) else None
        if not c or (c[0] == 0 and c[2] == 0):
            line_chars += txt[i]
            continue

        y, h, x = c[2], c[3] - c[2], c[0]

        if line_y == -1:
            line_y, line_h, line_x = y, h, x
            line_chars += txt[i]
            height_sum += h
            char_count += 1
        elif abs(y - line_y) > 200:
            gap = y - line_y
            lines.append({"text": line_chars.strip(), "y": line_y, "h": line_h, "x": line_x, "gap_after": gap})
            line_chars = txt[i]
            line_y, line_h, line_x = y, h, x
            height_sum += h
            char_count += 1
        else:
            if x < line_x:
                line_x = x
            line_chars += txt[i]
            height_sum += h
            char_count += 1

    if line_chars.strip():
        lines.append({"text": line_chars.strip(), "y": line_y, "h": line_h, "x": line_x, "gap_after": 0})

    if not lines:
        return ""

    body_h = round(height_sum / char_count) if char_count > 0 else 1452
    para_gap = body_h * 2

    # Merge continuation lines
    i = len(lines) - 1
    while i > 0:
        prev = lines[i - 1]
        cur = lines[i]
        if prev["text"] and cur["text"]:
            if prev["gap_after"] <= para_gap and abs(prev["h"] - cur["h"]) <= 200:
                if re.search(r"[a-zäöüß]$", prev["text"], re.I) and re.match(r"[a-zäöüß]", cur["text"]):
                    prev["text"] = prev["text"] + " " + cur["text"]
                    prev["gap_after"] = cur["gap_after"]
                    lines.pop(i)
        i -= 1

    # Build output
    parts = []
    for i, line in enumerate(lines):
        if not line["text"]:
            continue

        is_para = i > 0 and lines[i - 1]["gap_after"] > para_gap
        if is_para and parts:
            parts.append("")

        cleaned = clean_text(line["text"])
        if not cleaned:
            continue

        if not markdown:
            parts.append(cleaned)
            continue

        # List detection
        if re.match(r"^[»›]\s*$", cleaned):
            bullet_text = ""
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt["text"]:
                    j += 1
                    continue
                nc = clean_text(nxt["text"])
                if not nc:
                    j += 1
                    continue
                if re.match(r"^[»›•·]", nc) or nxt["h"] > body_h * 1.35:
                    break
                bullet_text += (" " if bullet_text else "") + nc
                if nxt["gap_after"] > para_gap:
                    j += 1
                    break
                j += 1
            # Skip processed lines (modify i via lines index)
            if bullet_text:
                parts.append(f"- {bullet_text}")
            continue
        if re.match(r"^[»›]\s+", cleaned):
            parts.append(f"- {re.sub(r'^[»›]\\s*', '', cleaned)}")
            continue
        if re.match(r"^[•·]\s*", cleaned):
            parts.append(f"- {re.sub(r'^[•·]\\s*', '', cleaned)}")
            continue
        if re.match(r"^[a-z]\)\s", cleaned):
            parts.append(f"  - {cleaned}")
            continue
        if re.match(r"^\(\s*\d+\s*\)\s", cleaned):
            parts.append(f"    - {cleaned}")
            continue

        # Skip standalone page numbers
        if re.match(r"^\d{1,3}$", cleaned):
            continue

        # Heading detection by character height
        if line["h"] > body_h * 1.7:
            parts.append(f"## {cleaned}")
            continue
        if line["h"] > body_h * 1.35:
            parts.append(f"### {cleaned}")
            continue
        if line["h"] > body_h * 1.2:
            parts.append(f"#### {cleaned}")
            continue

        # TOC-like entry
        toc_m = re.match(r"^(.+?)\s+(\d{1,3})$", cleaned)
        if toc_m and len(cleaned) < 80:
            parts.append(f"- {toc_m.group(1)} — {toc_m.group(2)}")
            continue

        parts.append(cleaned)

    return "\n".join(parts)


# -- Find LibreOffice --
_soffice_cache = None

def find_soffice() -> str | None:
    global _soffice_cache
    if _soffice_cache is not None:
        return _soffice_cache or None

    candidates = []
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    elif sys.platform == "darwin":
        candidates = ["/Applications/LibreOffice.app/Contents/MacOS/soffice"]
    else:
        candidates = ["/usr/bin/soffice"]

    for p in candidates:
        if Path(p).exists():
            _soffice_cache = p
            return p

    import shutil as _shutil
    if _shutil.which("soffice"):
        _soffice_cache = "soffice"
        return "soffice"

    _soffice_cache = ""
    return None


# -- Convert buffer to text --
def buffer_to_text(buf: bytes, ext: str, tmp_dir: Path) -> str | None:
    tmp_path = tmp_dir / f"_convert.{ext}"
    try:
        tmp_path.write_bytes(buf)
        if ext in ("doc", "docx", "rtf"):
            if sys.platform == "darwin":
                try:
                    result = subprocess.run(
                        ["textutil", "-convert", "txt", "-stdout", str(tmp_path)],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.returncode == 0:
                        return result.stdout
                except Exception:
                    pass
            soffice = find_soffice()
            if soffice:
                try:
                    subprocess.run(
                        [soffice, "--headless", "--convert-to", "txt:Text",
                         "--outdir", str(tmp_dir), str(tmp_path)],
                        capture_output=True, timeout=30,
                    )
                    txt_path = tmp_path.with_suffix(".txt")
                    if txt_path.exists():
                        text = txt_path.read_text("utf-8")
                        txt_path.unlink(missing_ok=True)
                        return text
                except Exception:
                    pass
        if ext == "pdf":
            # Use PyMuPDF
            try:
                doc = fitz.open(str(tmp_path))
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                return text
            except Exception:
                pass
    except Exception:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)
    return None


# -- Convert material to markdown --
def convert_to_markdown(buf: bytes, file_name: str, out_dir: Path) -> bool:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    text = buffer_to_text(buf, ext, out_dir)
    if not text:
        return False
    md_name = re.sub(r"\.[^.]+$", ".md", file_name)
    (out_dir / md_name).write_text(text, encoding="utf-8")
    return True


# -- Find a Unicode font for text overlay --
def find_unicode_font() -> str | None:
    if sys.platform == "win32":
        fonts = Path("C:\\Windows\\Fonts")
        candidates = [fonts / "arialuni.ttf", fonts / "arial.ttf", fonts / "segoeui.ttf"]
    else:
        candidates = [
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Helvetica.ttc"),
        ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# -- Main --
def main():
    args = sys.argv[1:]
    force = "--force" in args
    half_res = "--half-res" in args
    no_text = "--no-text" in args
    markdown = "--markdown" in args
    debug_text = "--debug-text" in args
    save_images = "--save-images" in args
    no_materials = "--no-materials" in args
    save_materials = not no_materials
    book_filter = int(args[args.index("--book") + 1]) if "--book" in args else None
    default_books = Path(__file__).resolve().parent / "books"
    output_dir = Path(args[args.index("--output") + 1]).resolve() if "--output" in args else default_books

    home = Path.home()
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        bibox_data = appdata / "BiBox 2.0"
    else:
        bibox_data = home / "Library" / "Application Support" / "BiBox 2.0"

    sync_dir = bibox_data / "synchronizedFiles"
    idb_blob_dir = bibox_data / "IndexedDB" / "app_angular_0.indexeddb.blob"
    ldb_dir = bibox_data / "IndexedDB" / "app_angular_0.indexeddb.leveldb"

    if not sync_dir.exists():
        print(f"BiBox synchronizedFiles nicht gefunden unter: {sync_dir}", file=sys.stderr, flush=True)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"BiBox-Daten: {bibox_data}", flush=True)
    print(f"Suche IndexedDB Blobs...", flush=True)
    blob_files = find_blob_files(idb_blob_dir)
    if not blob_files:
        print("Keine IndexedDB Blobs mit Buchdaten gefunden.", file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"  {len(blob_files)} Blob(s) gefunden.", flush=True)

    # Load all page text data
    print("Lade Text-Daten...", flush=True)
    all_page_data = find_all_page_data(sync_dir, ldb_dir)
    if all_page_data:
        for pd in all_page_data:
            ids = list(pd.keys())
            with_text = sum(1 for k in ids if pd[k].get("txt"))
            print(f"  Text-Daten gefunden: {with_text}/{len(ids)} Seiten.", flush=True)
    else:
        print("  Keine Text-Daten gefunden. PDF wird nicht durchsuchbar.", flush=True)

    # Find font
    font_path = None
    if not no_text and all_page_data:
        font_path = find_unicode_font()
        if font_path:
            print(f"  Font: {font_path}", flush=True)
        else:
            print("  Warnung: Kein Unicode-Font gefunden. Einige Zeichen könnten fehlen.", flush=True)

    # Extract book titles and persist cache
    book_titles = extract_book_titles(ldb_dir, output_dir)
    try:
        import json
        titles_cache = output_dir / "titles.json"
        titles_cache.write_text(
            json.dumps({str(k): v for k, v in book_titles.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    print("\nGefundene Bücher:", flush=True)
    for blob_path in blob_files:
        bid = extract_book_id(blob_path)
        title = book_titles.get(bid, f"Book {bid}")
        print(f"  {title} (ID {bid})", flush=True)

    for blob_path in blob_files:
        book_id = extract_book_id(blob_path)
        if book_filter and book_id != book_filter:
            continue

        book_title = book_titles.get(book_id)
        book_label = f"{book_title} ({book_id})" if book_title else f"Book {book_id}"
        pages = extract_page_mapping(blob_path, ldb_dir, book_id)

        if not pages:
            print(f"\n{book_label}: keine Seiten gefunden, überspringe.", flush=True)
            continue

        existing = [p for p in pages if hash_to_file_path(sync_dir, p["hash"]).exists()]
        print(f"\n{book_label}: {len(pages)} Seiten, {len(existing)} lokal verfügbar", flush=True)

        if not existing:
            print("  Keine lokalen Dateien, überspringe.", flush=True)
            continue

        # Split into main book (image) pages and solution (PDF) pages.
        # BiBox stores solution pages twice: as low-res JPEG previews AND full-quality PDFs.
        # Detect the contiguous JPEG block immediately before the PDF block → skip those as previews.
        def _quick_fmt(h: str) -> str:
            try:
                raw4 = hash_to_file_path(sync_dir, h).read_bytes()[:4]
                dec4 = decrypt(raw4)
                if dec4[:4] == b"%PDF":
                    return "pdf"
                if dec4[:2] == b"\xff\xd8":
                    return "jpg"
                if dec4[:2] == b"\x89\x50":
                    return "png"
            except Exception:
                pass
            return "other"

        fmts = [_quick_fmt(p["hash"]) for p in existing]
        first_pdf_idx = next((i for i, f in enumerate(fmts) if f == "pdf"), None)

        sol_pages: list[dict] = []
        if first_pdf_idx is not None:
            sol_pages = existing[first_pdf_idx:]
            # Find contiguous JPEG block immediately before PDF block = low-res previews.
            # Skip any non-jpg/non-pdf material (e.g. ZIP files) that may sit between the
            # JPEG previews and the solution PDFs, then count the trailing JPEG run.
            # Cap by number of solution PDFs (one preview per file).
            j = first_pdf_idx - 1
            while j >= 0 and fmts[j] not in ("jpg", "pdf"):
                j -= 1
            k = j
            while k >= 0 and fmts[k] == "jpg":
                k -= 1
            preview_count = min(j - k, len(sol_pages))
            preview_start = j + 1 - preview_count
            if preview_count:
                print(f"  {preview_count} Vorschau-Seiten übersprungen (als PDF verfügbar)", flush=True)
            existing = existing[:preview_start]  # main book pages only

        # Trim supplementary material images (screenshots, GeoGebra etc.) that appear
        # after the main book pages but before the solution previews/ZIPs/PDFs.
        # Heuristic: main book = initial contiguous block with the same format.
        # Only applies when the book starts with PNG (scanned pages), since PNG→JPG
        # transitions reliably mark the boundary between book pages and supplements.
        if existing and fmts[0] == "png":
            main_end = next(
                (i for i in range(len(existing)) if fmts[i] != "png"),
                len(existing),
            )
            if 0 < main_end < len(existing):
                print(f"  {len(existing) - main_end} Zusatzmaterial-Seiten übersprungen", flush=True)
                existing = existing[:main_end]

        # Match pageData by page count
        page_data_map = None
        sorted_page_ids = None
        if all_page_data:
            max_blob_page = max(p["page"] for p in pages) if pages else 0
            best = min(all_page_data, key=lambda pd: abs(len(pd) - max_blob_page))
            sorted_ids = sorted(int(k) for k in best.keys())
            if len(sorted_ids) >= max_blob_page:
                page_data_map = best
                sorted_page_ids = sorted_ids

        # Output directory
        base_name = re.sub(r'[/\\:*?"<>|]', "-", book_title) if book_title else f"book-{book_id}"
        dir_name = f"{base_name} ({book_id})" if book_title else base_name
        book_dir = output_dir / dir_name
        pdf_path = book_dir / f"{base_name}.pdf"

        if not force and pdf_path.exists():
            print("  Bereits vorhanden, überspringe. (--force zum Überschreiben)", flush=True)
            continue

        book_dir.mkdir(parents=True, exist_ok=True)

        # Build PDF with PyMuPDF
        overlay_mode = "nur Bilder" if no_text else "mit Text-Overlay"
        print(f"  PDF erstellen ({overlay_mode})...", flush=True)

        pdf_doc = fitz.open()
        count = 0

        # Load font ONCE for all pages
        overlay_font = None
        if not no_text and font_path:
            try:
                overlay_font = fitz.Font(fontfile=font_path)
            except Exception as e:
                print(f"  Font laden fehlgeschlagen: {e}", flush=True)

        overlay_color = (1, 0, 0) if debug_text else (0, 0, 0)
        overlay_opacity = 0.5 if debug_text else 0

        for p in existing:
            file_path = hash_to_file_path(sync_dir, p["hash"])

            encrypted = file_path.read_bytes()
            decrypted = decrypt(encrypted)

            # Detect image format
            if decrypted[:2] == b"\xff\xd8":
                ext = "jpg"
            elif decrypted[:2] == b"\x89\x50":
                ext = "png"
            elif decrypted[:4] == b"%PDF":
                ext = "pdf"
            else:
                continue

            # Save individual image
            if save_images:
                img_dir = book_dir / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / f"page-{p['page']:04d}.{ext}").write_bytes(decrypted)

            try:
                # Create page from image, scaled to correct physical size.
                # Read pixel dims from header (not img_page.rect — PyMuPDF scales that internally).
                pixel_w, pixel_h = _pixel_dims(decrypted, ext)
                dpi = _image_dpi(decrypted, ext) or 300.0
                if half_res and ext in ("png", "jpg"):
                    pix = fitz.Pixmap(decrypted)
                    if pix.alpha:  # JPEG doesn't support transparency — drop alpha channel
                        pix = fitz.Pixmap(pix, 0)
                    pix.shrink(1)  # halves both dimensions in-place
                    decrypted = pix.tobytes("jpeg", jpg_quality=85)
                    ext = "jpg"
                    pixel_w, pixel_h = pix.width, pix.height
                    dpi /= 2  # same physical page size, half the stored resolution
                scale = 72.0 / dpi
                page_w = pixel_w * scale
                page_h = pixel_h * scale
                pdf_page = pdf_doc.new_page(width=page_w, height=page_h)
                pdf_page.insert_image(pdf_page.rect, stream=decrypted)

                # Add text overlay (one TextWriter per page, font reused)
                if overlay_font and sorted_page_ids and 1 <= p["page"] <= len(sorted_page_ids):
                    page_id = sorted_page_ids[p["page"] - 1]
                    pd = page_data_map.get(str(page_id))
                    if pd and pd.get("txt") and pd.get("cds"):
                        words = extract_words(pd["txt"], pd["cds"])
                        tw = fitz.TextWriter(pdf_page.rect)
                        appended = 0
                        for w in words:
                            x = (w["x"] / 100) * page_w
                            y = (w["y"] / 100) * page_h
                            target_w = (w["w"] / 100) * page_w
                            target_h = (w["h"] / 100) * page_h

                            # Calculate font size from target width (like JS version)
                            width_at_1 = overlay_font.text_length(w["text"], fontsize=1)
                            if width_at_1 <= 0:
                                continue
                            font_size = min(target_w / width_at_1, target_h)
                            if font_size < 0.5:
                                continue

                            try:
                                tw.append(fitz.Point(x, y + target_h), w["text"],
                                         fontsize=font_size, font=overlay_font)
                                appended += 1
                            except Exception:
                                pass
                        if appended:
                            tw.write_text(pdf_page, color=overlay_color, opacity=overlay_opacity)

                count += 1
                if count % 50 == 0:
                    print(f"  {count}/{len(existing)} Seiten...", end="\r", flush=True)
            except Exception as e:
                print(f"  Seite {p['page']}: Fehler: {e}", flush=True)
                continue

        print(f"  PDF speichern...", end="", flush=True)
        pdf_doc.save(str(pdf_path), garbage=4, deflate=True)
        size_mb = pdf_path.stat().st_size / 1024 / 1024
        pdf_doc.close()
        print(f" {count} Seiten, {size_mb:.1f} MB -> {pdf_path}", flush=True)

        # Solutions PDF (separate file for full-quality PDF pages)
        if sol_pages:
            sol_path = book_dir / f"{base_name} - Lösungen.pdf"
            print(f"  Lösungen als separate PDF...", flush=True)
            sol_doc = fitz.open()
            sol_count = 0
            for sp in sol_pages:
                try:
                    enc = hash_to_file_path(sync_dir, sp["hash"]).read_bytes()
                    dec = decrypt(enc)
                    if dec[:4] != b"%PDF":
                        continue
                    sub = fitz.open(stream=dec, filetype="pdf")
                    n = sub.page_count
                    for sub_page in sub:
                        rect = sub_page.rect
                        sol_page = sol_doc.new_page(width=rect.width, height=rect.height)
                        sol_page.show_pdf_page(rect, sub, sub_page.number)
                    sub.close()
                    sol_count += n
                except Exception as e:
                    print(f"  Lösung {sp['page']}: Fehler: {e}", flush=True)
            sol_doc.save(str(sol_path), garbage=4, deflate=True)
            sol_size_mb = sol_path.stat().st_size / 1024 / 1024
            sol_doc.close()
            print(f"  -> {sol_count} Seiten, {sol_size_mb:.1f} MB -> {sol_path}", flush=True)

        # Export text
        if sorted_page_ids and page_data_map:
            ext = "md" if markdown else "txt"
            print(f"  Text exportieren ({ext})...", flush=True)
            content = ""

            for p in pages:
                page_id = sorted_page_ids[p["page"] - 1] if 1 <= p["page"] <= len(sorted_page_ids) else None
                pd = page_data_map.get(str(page_id)) if page_id is not None else None

                if markdown:
                    content += (format_page_text(pd["txt"], pd["cds"], markdown=True) if pd and pd.get("txt") else "") + "\n\n"
                else:
                    content += (pd.get("txt", "") if pd else "") + "\n\n"

            out_path = book_dir / f"{base_name}.{ext}"
            out_path.write_text(content, encoding="utf-8")
            print(f"  -> {out_path}", flush=True)

        # Export materials
        print("  Zusatzmaterial-Referenz erstellen...", flush=True)
        blob_buf = blob_path.read_bytes()
        materials = extract_materials(blob_buf)

        if materials:
            seen = {}
            unique = []
            for m in materials:
                key = m["title"] + m["file"]
                if key not in seen:
                    seen[key] = True
                    unique.append(m)

            by_ext: dict[str, list] = {}
            for m in unique:
                by_ext.setdefault(m["ext"], []).append(m)

            md = f"# Zusatzmaterial — {book_title or f'Book {book_id}'}\n\n"
            md += f"Insgesamt {len(unique)} Dateien.\n\n"
            for ext_name, items in sorted(by_ext.items()):
                md += f"## {ext_name.upper()} ({len(items)})\n\n"
                for m in items:
                    md += f"- {m['title']} — `{m['file']}`\n"
                md += "\n"

            (book_dir / "Zusatzmaterial.md").write_text(md, encoding="utf-8")
            print(f"  -> {book_dir / 'Zusatzmaterial.md'} ({len(unique)} Einträge)", flush=True)

            # Download and convert materials
            if save_materials and any(m["md5sum"] for m in unique):
                mat_dir = book_dir / "Zusatzmaterial"
                mat_dir.mkdir(parents=True, exist_ok=True)
                saved = 0
                converted = 0
                skipped = 0

                for m in unique:
                    if not m["md5sum"]:
                        skipped += 1
                        continue
                    file_path = hash_to_file_path(sync_dir, m["md5sum"])
                    if not file_path.exists():
                        skipped += 1
                        continue

                    decrypted = decrypt(file_path.read_bytes())
                    out_name = re.sub(r'[/\\:*?"<>|]', "-", m["file"])

                    (mat_dir / out_name).write_bytes(decrypted)
                    saved += 1

                    if convert_to_markdown(decrypted, out_name, mat_dir):
                        converted += 1

                print(f"  Materialien gespeichert: {saved} Dateien -> {mat_dir}", flush=True)
                if converted:
                    print(f"  Materialien konvertiert: {converted} Markdown-Dateien", flush=True)
                if skipped:
                    print(f"  {skipped} übersprungen (kein Hash oder nicht lokal)", flush=True)
        else:
            print("  Kein Zusatzmaterial gefunden.", flush=True)

    print("\nFertig.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFehler: {e}", file=sys.stderr, flush=True)
    if getattr(sys, "frozen", False):
        input("\nDrücke Enter zum Beenden...")
