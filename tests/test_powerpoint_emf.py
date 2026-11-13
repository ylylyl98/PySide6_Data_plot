import struct
import unittest
from io import BytesIO
from unittest.mock import MagicMock

from PIL import Image
from ui_qt.quick.powerpoint_emf import UnsupportedEmf, extract_emf_png, export_shape_png


def _png():
    stream = BytesIO()
    Image.new('RGB', (3, 2), (17, 34, 51)).save(stream, format='PNG')
    return stream.getvalue()


def _record(kind, flags, payload):
    padding = b'\0' * (-len(payload) % 4)
    return struct.pack('<HHII', kind, flags, 12 + len(payload) + len(padding), len(payload)) + payload + padding


def _emf(records):
    comment_data = b'EMF+' + b''.join(records)
    comment = struct.pack('<III', 70, 12 + len(comment_data), len(comment_data)) + comment_data
    end = struct.pack('<IIIII', 14, 20, 0, 0, 20)
    header = bytearray(88)
    struct.pack_into('<II', header, 0, 1, 88)
    header[40:44] = b' EMF'
    struct.pack_into('<I', header, 48, len(header) + len(comment) + len(end))
    return bytes(header) + comment + end


def _object(png):
    data = struct.pack('<IIIIIII', 0xDBC01002, 1, 0, 0, 0, 0, 1) + png
    return data + b'\0' * (-len(data) % 4)


class EmfParserTests(unittest.TestCase):
    def test_single_png_image(self):
        png = _png()
        self.assertEqual(extract_emf_png(_emf([_record(0x4008, 0x0501, _object(png))])), png)

    def test_continuation_and_powerpoint_last_continuation_flag(self):
        png = _png()
        obj = _object(png)
        for last_flags, last_payload in ((0x0501, obj[40:]), (0x8501, struct.pack('<I', len(obj)) + obj[40:])):
            records = [_record(0x4008, 0x8501, struct.pack('<I', len(obj)) + obj[:40]),
                       _record(0x4008, last_flags, last_payload)]
            self.assertEqual(extract_emf_png(_emf(records)), png)

    def test_rejects_truncated_continuation_wrong_id_and_multiple_images(self):
        obj = _object(_png())
        first = _record(0x4008, 0x8501, struct.pack('<I', len(obj)) + obj[:40])
        wrong_id = _record(0x4008, 0x8502, struct.pack('<I', len(obj)) + obj[40:])
        full = _record(0x4008, 0x0501, obj)
        for records in ([first], [first, wrong_id], [full, full]):
            with self.subTest(records=len(records)), self.assertRaises(UnsupportedEmf):
                extract_emf_png(_emf(records))

    def test_rejects_bad_header_bad_crc_and_unsupported_bitmap(self):
        png = _png()
        corrupt = bytearray(png); corrupt[20] ^= 1
        bad_type = bytearray(_object(png)); struct.pack_into('<I', bad_type, 24, 0)
        invalid = bytearray(_emf([_record(0x4008, 0x0501, _object(png))])); invalid[40] = 0
        for raw in (bytes(invalid), _emf([_record(0x4008, 0x0501, _object(corrupt))]), _emf([_record(0x4008, 0x0501, bytes(bad_type))])):
            with self.assertRaises(UnsupportedEmf):
                extract_emf_png(raw)

    def test_export_preserves_shape_and_returns_candidate(self):
        png = _png()
        data = _emf([_record(0x4008, 0x0501, _object(png))])
        shape = MagicMock()
        shape.Id, shape.Type = 3, 13
        shape.Left, shape.Top, shape.Width, shape.Height, shape.Rotation = 1, 2, 3, 4, 0
        slide = MagicMock()
        slide.Shapes.Count = 1
        slide.Shapes.return_value = shape
        slide.Parent.Saved = 0
        shape.Parent = slide
        def export(path, kind):
            with open(path, 'wb') as stream:
                stream.write(data)
        shape.Export.side_effect = export
        self.assertEqual(export_shape_png(shape), png)
