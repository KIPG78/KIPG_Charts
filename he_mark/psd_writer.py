"""Streaming writer for layered 8-bit RGB Photoshop (.psd) files.

Built for one job: flat-colour vector artwork at very large pixel sizes, where
the picture cannot be held in memory.  Every layer is one flat colour plus a
coverage (alpha) channel, so:

  * the R, G and B channels of a layer are constant, and their compressed
    rows are computed once and repeated;
  * only the alpha rows carry information, and they arrive band by band;
  * everything is PackBits (RLE) compressed, the one compression every PSD
    reader understands.

What goes in the file (the things a careful reader looks for):

  header            8BPS, version 1, 4 channels (RGB + transparency), 8 bit
  resources         resolution (1005), JPEG thumbnail (1036), sRGB ICC profile
                    (1039), version info with "has real merged data" (1057)
  layers            bottom to top, cropped to their bounds, channel order
                    [-1, 0, 1, 2], ASCII + Unicode names, group folders (lsct)
  merged image      the flattened picture, matted on white the way Photoshop
                    writes it, with the transparency channel fourth

PSD version 1 tops out at 30,000 px per side; this writer refuses to go over.
"""
from __future__ import annotations

import os
import struct
import tempfile

import numpy as np

PSD_MAX_SIDE = 30000


# --------------------------------------------------------------------------
# PackBits
# --------------------------------------------------------------------------
def packbits_const(value: int, width: int) -> bytes:
    """One row of `width` identical bytes."""
    full, rem = divmod(width, 128)
    out = bytes((129, value)) * full          # 129 == -127: repeat next byte 128 times
    if rem == 1:
        out += bytes((0, value))              # literal run of one
    elif rem > 1:
        out += bytes((257 - rem, value))      # repeat `rem` times
    return out


def packbits_row(row: np.ndarray) -> bytes:
    """PackBits-encode one row of uint8."""
    n = row.shape[0]
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    if change.size == 0:
        return packbits_const(int(row[0]), n)
    starts = np.concatenate(([0], change)).tolist()
    ends = change.tolist() + [n]
    raw = row.tobytes()
    out = bytearray()
    lit_from = -1  # start index of a pending literal stretch, or -1

    def flush(upto: int) -> None:
        nonlocal lit_from
        if lit_from < 0:
            return
        i = lit_from
        while i < upto:
            k = min(128, upto - i)
            out.append(k - 1)
            out.extend(raw[i:i + k])
            i += k
        lit_from = -1

    for s, e in zip(starts, ends):
        length = e - s
        if length >= 3:
            flush(s)
            out.extend(packbits_const(raw[s], length))
        elif lit_from < 0:
            lit_from = s
    flush(n)
    return bytes(out)


def unpackbits(data: bytes, width: int) -> bytes:
    """Reference decoder, used by the self-test."""
    out = bytearray()
    i = 0
    while len(out) < width:
        h = data[i]
        i += 1
        if h < 128:
            out.extend(data[i:i + h + 1])
            i += h + 1
        elif h > 128:
            out.extend(bytes((data[i],)) * (257 - h))
            i += 1
    return bytes(out)


# --------------------------------------------------------------------------
# small binary helpers
# --------------------------------------------------------------------------
def _pascal(name: str, pad: int) -> bytes:
    raw = name.encode("mac_roman", "replace")[:255]
    b = bytes((len(raw),)) + raw
    return b + b"\0" * (-len(b) % pad)


def _unicode(text: str) -> bytes:
    return struct.pack(">I", len(text)) + text.encode("utf-16-be")


def _resource(rid: int, data: bytes) -> bytes:
    return b"8BIM" + struct.pack(">H", rid) + b"\0\0" + struct.pack(">I", len(data)) + data + b"\0" * (len(data) % 2)


def _tagged(key: bytes, data: bytes) -> bytes:
    data = data + b"\0" * (-len(data) % 4)
    return b"8BIM" + key + struct.pack(">I", len(data)) + data


def _srgb_icc() -> bytes:
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
class Layer:
    """One flat-colour pixel layer.  Feed alpha rows top to bottom."""

    def __init__(self, name: str, rgb: tuple[int, int, int], bbox: tuple[int, int, int, int],
                 tmpdir: str, opacity: int = 255, visible: bool = True):
        self.name = name
        self.rgb = tuple(int(v) for v in rgb)
        self.left, self.top, self.right, self.bottom = (int(v) for v in bbox)
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.opacity = opacity
        self.visible = visible
        self.kind = "pixel"
        self._counts: list[int] = []
        self._tmp = tempfile.NamedTemporaryFile(dir=tmpdir, prefix="alpha_", delete=False)
        self._bytes = 0

    def add_encoded_rows(self, rows: list[bytes]) -> None:
        for r in rows:
            self._counts.append(len(r))
            self._tmp.write(r)
            self._bytes += len(r)

    def close(self) -> None:
        self._tmp.close()
        if len(self._counts) != self.height:
            raise ValueError(f"layer {self.name!r}: got {len(self._counts)} rows, expected {self.height}")

    # -- sizes -------------------------------------------------------------
    def channel_lengths(self) -> list[tuple[int, int]]:
        h = self.height
        const = len(packbits_const(0, self.width)) if self.width else 0
        return [(-1, 2 + 2 * h + self._bytes)] + [(c, 2 + 2 * h + const * h) for c in (0, 1, 2)]

    def record(self) -> bytes:
        chans = self.channel_lengths()
        rec = struct.pack(">iiii", self.top, self.left, self.bottom, self.right)
        rec += struct.pack(">H", len(chans))
        for cid, length in chans:
            rec += struct.pack(">hI", cid, length)
        flags = 0x08 | (0 if self.visible else 0x02)
        rec += b"8BIM" + b"norm" + bytes((self.opacity, 0, flags, 0))
        extra = struct.pack(">I", 0)                              # no layer mask
        ranges = bytes((0, 0, 255, 255, 0, 0, 255, 255)) * (len(chans) + 1)
        extra += struct.pack(">I", len(ranges)) + ranges          # default blend-if ranges
        extra += _pascal(self.name, 4)
        extra += _tagged(b"luni", _unicode(self.name))
        return rec + struct.pack(">I", len(extra)) + extra

    def write_channels(self, f) -> None:
        h, w = self.height, self.width
        f.write(struct.pack(">H", 1))                              # RLE
        f.write(struct.pack(f">{h}H", *self._counts))
        with open(self._tmp.name, "rb") as src:
            while True:
                chunk = src.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        for value in self.rgb:
            row = packbits_const(value, w)
            f.write(struct.pack(">H", 1))
            f.write(struct.pack(f">{h}H", *([len(row)] * h)))
            step = max(1, (1 << 22) // max(1, len(row)))
            done = 0
            while done < h:
                k = min(step, h - done)
                f.write(row * k)
                done += k
        os.unlink(self._tmp.name)


class GroupMarker:
    """Folder open/close records.  `kind` is 'open' (the named folder, written
    after its children) or 'close' (the hidden divider written before them)."""

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.height = self.width = 0

    def record(self) -> bytes:
        rec = struct.pack(">iiii", 0, 0, 0, 0) + struct.pack(">H", 4)
        for cid in (-1, 0, 1, 2):
            rec += struct.pack(">hI", cid, 2)
        if self.kind == "open":
            name, divider, flags = self.name, struct.pack(">I", 1) + b"8BIM" + b"pass", 0x08
        else:
            name, divider, flags = "</Layer group>", struct.pack(">I", 3), 0x08 | 0x10
        blend = b"pass" if self.kind == "open" else b"norm"
        rec += b"8BIM" + blend + bytes((255, 0, flags, 0))
        ranges = bytes((0, 0, 255, 255, 0, 0, 255, 255)) * 5
        extra = struct.pack(">I", 0) + struct.pack(">I", len(ranges)) + ranges
        extra += _pascal(name, 4)
        extra += _tagged(b"luni", _unicode(name))
        extra += _tagged(b"lsct", divider)
        return rec + struct.pack(">I", len(extra)) + extra

    def write_channels(self, f) -> None:
        f.write(struct.pack(">H", 0) * 4)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------
class PsdDocument:
    def __init__(self, width: int, height: int, ppi: float = 300.0, tmpdir: str | None = None):
        if max(width, height) > PSD_MAX_SIDE:
            raise ValueError(f"PSD allows at most {PSD_MAX_SIDE} px per side, got {width}x{height}")
        self.width, self.height, self.ppi = width, height, ppi
        self.tmpdir = tmpdir or tempfile.mkdtemp(prefix="psd_")
        self.items: list = []          # bottom-to-top, exactly the order written
        self._merged = [tempfile.NamedTemporaryFile(dir=self.tmpdir, prefix=f"merged{c}_", delete=False)
                        for c in range(4)]
        self._merged_counts: list[list[int]] = [[] for _ in range(4)]

    def add_layer(self, name, rgb, bbox, **kw) -> Layer:
        layer = Layer(name, rgb, bbox, self.tmpdir, **kw)
        self.items.append(layer)
        return layer

    def begin_group(self) -> None:
        """Call before adding the group's children (bottom-to-top order)."""
        self.items.append(GroupMarker("", "close"))

    def end_group(self, name: str) -> None:
        self.items.append(GroupMarker(name, "open"))

    def add_merged_rows(self, channel: int, rows: list[bytes]) -> None:
        for r in rows:
            self._merged_counts[channel].append(len(r))
            self._merged[channel].write(r)

    def set_thumbnail(self, jpeg: bytes, width: int, height: int) -> None:
        """Finder / Quick Look thumbnail (image resource 1036, JPEG RGB)."""
        row_bytes = (width * 24 + 31) // 32 * 4
        self._thumb = (struct.pack(">IIIIIIHH", 1, width, height, row_bytes, row_bytes * height,
                                   len(jpeg), 24, 1) + jpeg)

    def save(self, path: str, writer_name: str = "he_mark") -> int:
        for item in self.items:
            item.close()
        for t in self._merged:
            t.close()
        for c in range(4):
            if len(self._merged_counts[c]) != self.height:
                raise ValueError(f"merged channel {c}: {len(self._merged_counts[c])} rows, expected {self.height}")

        res = b""
        fixed = int(round(self.ppi * 65536))
        res += _resource(1005, struct.pack(">IHHIHH", fixed, 1, 1, fixed, 1, 1))
        if getattr(self, "_thumb", None):
            res += _resource(1036, self._thumb)
        res += _resource(1039, _srgb_icc())
        res += _resource(1057, struct.pack(">IB", 1, 1) + _unicode(writer_name) + _unicode(writer_name)
                         + struct.pack(">I", 1))

        records = b"".join(i.record() for i in self.items)
        chan_bytes = 0
        for i in self.items:
            if isinstance(i, Layer):
                chan_bytes += sum(length for _, length in i.channel_lengths())
            else:
                chan_bytes += 8
        layer_info_len = 2 + len(records) + chan_bytes
        pad = layer_info_len % 2
        layer_info_len += pad
        section_len = 4 + layer_info_len + 4            # layer info + empty global mask

        with open(path, "wb") as f:
            f.write(b"8BPS" + struct.pack(">H", 1) + b"\0" * 6)
            f.write(struct.pack(">HIIHH", 4, self.height, self.width, 8, 3))
            f.write(struct.pack(">I", 0))                           # colour mode data
            f.write(struct.pack(">I", len(res)) + res)
            f.write(struct.pack(">I", section_len))
            f.write(struct.pack(">I", layer_info_len))
            # negative count: the merged image's first extra channel is transparency
            f.write(struct.pack(">h", -len(self.items)))
            f.write(records)
            for i in self.items:
                i.write_channels(f)
            f.write(b"\0" * pad)
            f.write(struct.pack(">I", 0))                           # global layer mask info
            f.write(struct.pack(">H", 1))                           # merged image: RLE
            for c in range(4):
                f.write(struct.pack(f">{self.height}H", *self._merged_counts[c]))
            for c in range(4):
                with open(self._merged[c].name, "rb") as src:
                    while True:
                        chunk = src.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                os.unlink(self._merged[c].name)
            size = f.tell()
        return size
