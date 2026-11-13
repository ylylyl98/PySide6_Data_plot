"""Strict, best-effort EMF+ PNG extraction; a result is only a candidate.

Office can render or transform images while exporting. Callers must compare the
candidate byte-for-byte against independently trusted PNG data; fall back to a
fresh PowerPoint snapshot when those bytes are unavailable or do not match.
"""
from io import BytesIO
from pathlib import Path
import struct
import zlib

from PIL import Image
from ui_qt.quick.powerpoint_temp import temporary_directory

MAX_EMF = 128 * 1024 * 1024
MAX_IMAGE_OBJECT = 64 * 1024 * 1024
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


class UnsupportedEmf(ValueError):
    pass


def _png_from_image_object(data):
    if len(data) < 36:
        raise UnsupportedEmf('Truncated EMF+ image object')
    version, image_type = struct.unpack_from('<II', data, 0)
    bitmap_type = struct.unpack_from('<I', data, 24)[0]
    if version >> 12 != 0xDBC01 or image_type != 1 or bitmap_type != 1:
        raise UnsupportedEmf('Unsupported EMF+ image representation')
    png = data[28:]
    if not png.startswith(_PNG_SIGNATURE):
        raise UnsupportedEmf('EMF+ image is not a PNG')
    pos = 8
    first = True
    has_data = False
    while pos + 12 <= len(png):
        length = struct.unpack_from('>I', png, pos)[0]
        end = pos + 12 + length
        if length > MAX_IMAGE_OBJECT or end > len(png):
            raise UnsupportedEmf('Truncated PNG chunk')
        kind = png[pos + 4:pos + 8]
        if first and (kind != b'IHDR' or length != 13):
            raise UnsupportedEmf('PNG has no valid header')
        first = False
        if zlib.crc32(png[pos + 4:pos + 8 + length]) & 0xffffffff != struct.unpack_from('>I', png, pos + 8 + length)[0]:
            raise UnsupportedEmf('PNG chunk checksum mismatch')
        has_data |= kind == b'IDAT'
        pos = end
        if kind == b'IEND':
            if length != 0 or not has_data or len(png) - pos > 3 or any(png[pos:]):
                raise UnsupportedEmf('Invalid PNG terminator or object padding')
            result = png[:pos]
            try:
                with Image.open(BytesIO(result)) as image:
                    if image.format != 'PNG':
                        raise ValueError('Not PNG')
                    image.verify()
            except Exception as exc:
                raise UnsupportedEmf('Invalid PNG image') from exc
            return result
    raise UnsupportedEmf('PNG terminator not found')


def extract_emf_png(data):
    """Extract exactly one compressed PNG image object from an EMF+ file."""
    if len(data) < 88 or len(data) > MAX_EMF:
        raise UnsupportedEmf('Unsupported EMF size')
    if struct.unpack_from('<I', data, 0)[0] != 1 or struct.unpack_from('<I', data, 4)[0] < 88 or data[40:44] != b' EMF':
        raise UnsupportedEmf('Invalid EMF header')
    declared = struct.unpack_from('<I', data, 48)[0]
    if declared != len(data):
        raise UnsupportedEmf('Invalid EMF length')
    offset = 0
    objects = []
    active = None
    saw_eof = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise UnsupportedEmf('Truncated EMF record')
        kind, size = struct.unpack_from('<II', data, offset)
        if size < 8 or size % 4 or offset + size > len(data):
            raise UnsupportedEmf('Invalid EMF record size')
        if kind == 70:
            if size < 12:
                raise UnsupportedEmf('Truncated EMF comment')
            count = struct.unpack_from('<I', data, offset + 8)[0]
            if count > size - 12:
                raise UnsupportedEmf('Invalid EMF comment size')
            comment = data[offset + 12:offset + 12 + count]
            if comment.startswith(b'EMF+'):
                pos = 4
                while pos < len(comment):
                    if pos + 12 > len(comment):
                        raise UnsupportedEmf('Truncated EMF+ record')
                    record_type, flags, record_size, data_size = struct.unpack_from('<HHII', comment, pos)
                    if record_size < 12 or record_size % 4 or pos + record_size > len(comment) or data_size > record_size - 12:
                        raise UnsupportedEmf('Invalid EMF+ record size')
                    payload = comment[pos + 12:pos + 12 + data_size]
                    object_type = (flags >> 8) & 0x7f
                    object_id = flags & 0xff
                    is_image = record_type == 0x4008 and object_type == 5
                    if active is not None and not is_image:
                        raise UnsupportedEmf('Image continuation was interrupted')
                    if is_image:
                        if object_id > 63:
                            raise UnsupportedEmf('Invalid EMF+ image object ID')
                        continued = bool(flags & 0x8000)
                        if active is not None and object_id != active[0]:
                            raise UnsupportedEmf('EMF+ image object ID changed')
                        if continued:
                            if len(payload) < 4:
                                raise UnsupportedEmf('Missing EMF+ total object size')
                            total = struct.unpack_from('<I', payload, 0)[0]
                            if total < 36 or total > MAX_IMAGE_OBJECT:
                                raise UnsupportedEmf('Unsupported EMF+ image size')
                            if active is None:
                                active = [object_id, total, bytearray()]
                            elif total != active[1]:
                                raise UnsupportedEmf('EMF+ total object size changed')
                            active[2].extend(payload[4:])
                        elif active is not None:
                            active[2].extend(payload)
                        else:
                            if len(payload) > MAX_IMAGE_OBJECT:
                                raise UnsupportedEmf('Unsupported EMF+ image size')
                            objects.append(bytes(payload))
                        if active is not None:
                            if len(active[2]) > active[1]:
                                raise UnsupportedEmf('EMF+ image exceeds its declared size')
                            if len(active[2]) == active[1]:
                                # Current PowerPoint can leave C set on the last
                                # fragment; the specification also defines the
                                # declared total as the continuation boundary.
                                objects.append(bytes(active[2])); active = None
                            elif not continued:
                                raise UnsupportedEmf('Premature final EMF+ image fragment')
                        if len(objects) > 1:
                            raise UnsupportedEmf('Multiple EMF+ image objects')
                    pos += record_size
        if kind == 14:
            saw_eof = True
            if offset + size != len(data):
                raise UnsupportedEmf('Data after EMF end record')
        offset += size
    if not saw_eof or active is not None or len(objects) != 1:
        raise UnsupportedEmf('Incomplete or ambiguous EMF+ image')
    return _png_from_image_object(objects[0])


def export_png_candidate(slide, shape_id):
    """Export a shape without clipboard use; returned PNG is not yet trusted."""
    shapes = slide.Shapes
    shape = next((shapes(i) for i in range(1, shapes.Count + 1) if int(shapes(i).Id) == int(shape_id)), None)
    if shape is None:
        raise UnsupportedEmf('Shape no longer exists')
    shape_type = int(shape.Type)
    if shape_type != 13 and not (shape_type == 14 and int(shape.PlaceholderFormat.ContainedType) == 13):
        raise UnsupportedEmf('Shape is not an embedded picture')
    pres = slide.Parent
    saved = int(pres.Saved)
    def identity(value):
        picture = value.PictureFormat
        return (int(value.Id), int(value.Type), *(float(getattr(value, name)) for name in ('Left', 'Top', 'Width', 'Height', 'Rotation')),
                *(float(getattr(picture, name)) for name in ('CropLeft', 'CropRight', 'CropTop', 'CropBottom')))
    before = identity(shape)
    slide_id, presentation_path = int(slide.SlideID), str(pres.FullName)
    with temporary_directory() as directory:
        filename = Path(directory) / 'shape.emf'
        shape.Export(str(filename), 5)
        if filename.stat().st_size > MAX_EMF:
            raise UnsupportedEmf('EMF export is too large')
        result = extract_emf_png(filename.read_bytes())
    shapes = slide.Shapes
    current = next((shapes(i) for i in range(1, shapes.Count + 1) if int(shapes(i).Id) == int(shape_id)), None)
    if current is None:
        raise UnsupportedEmf('Shape changed during export')
    after = identity(current)
    if before != after or int(pres.Saved) != saved or int(slide.SlideID) != slide_id or str(pres.FullName) != presentation_path:
        raise UnsupportedEmf('Presentation changed during export')
    return result


def export_shape_png(shape):
    """Return an untrusted PNG candidate, or raise UnsupportedEmf."""
    return export_png_candidate(shape.Parent, int(shape.Id))
