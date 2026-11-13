import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import unittest

from PIL import Image, PngImagePlugin
from core.pptx_figure_index import FigureIndex, read_shape_image


def png(text=None):
    output = BytesIO()
    info = PngImagePlugin.PngInfo()
    if text:
        info.add_text('note', text)
    Image.new('RGBA', (2, 3), (12, 34, 56, 255)).save(output, format='PNG', pnginfo=info)
    return output.getvalue()


class FigureIndexTests(unittest.TestCase):
    def test_known_export_sidecar_skips_history_but_validates_content(self):
        from unittest.mock import patch
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'Processed Data' / 'PL'
            folder.mkdir(parents=True)
            image = folder / 'sample_linear.png'
            image.write_bytes(png())
            data = folder / 'sample.dat'
            data.write_text('data')
            payload = {'operation': 'PL', 'outputs': [data.name, image.name],
                       'figures': {image.name: {'data_file': data.name, 'plot': {'title': 'current'}}}}
            metadata = folder / 'sample.metadata.json'
            metadata.write_text(json.dumps(payload))
            (folder / 'unrelated.metadata.json').write_text('malformed history')
            index = FigureIndex(tmp)
            with patch.object(index, '_metadata', wraps=index._metadata) as reads:
                records = index.find_png(image, metadata_path=metadata)
                self.assertEqual(records[0].plot['title'], 'current')
                reads.assert_called_once_with(metadata.resolve())
            self.assertEqual(records, FigureIndex(tmp).find_png(image))
            image.write_bytes(b'corrupted PNG')
            self.assertEqual(index.find_png(image, metadata_path=metadata), [])
            image.write_bytes(png())
            data.unlink()
            self.assertEqual(index.find_png(image, metadata_path=metadata), [])

    def test_known_sidecar_rejects_invalid_manifest_without_history_fallback(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'Processed Data' / 'PL'
            folder.mkdir(parents=True)
            image = folder / 'sample_linear.png'
            image.write_bytes(png())
            (folder / 'sample.dat').write_text('data')
            payload = {'operation': 'PL', 'outputs': ['sample.dat', image.name],
                       'figures': {image.name: {'data_file': 'sample.dat', 'plot': {}}}}
            good = folder / 'old.metadata.json'
            good.write_text(json.dumps(payload))
            selected = folder / 'new.metadata.json'
            index = FigureIndex(tmp)
            for bad in ('invalid json', '{}', json.dumps({**payload, 'display_save_pending': True}),
                        json.dumps({**payload, 'outputs': []})):
                selected.write_text(bad)
                self.assertEqual(index.find_png(image, metadata_path=selected), [])
            outside = Path(tmp) / 'outside.metadata.json'
            outside.write_text(json.dumps(payload))
            self.assertEqual(index.find_png(image, metadata_path=outside), [])
            self.assertEqual(len(index.find_png(image)), 1)

    def test_metadata_and_exact_pixels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'Processed Data' / 'PL'
            directory.mkdir(parents=True)
            (directory / 'plot_linear.dat').write_text('data')
            for name in ('plot_linear.png', 'plot_log.png'):
                (directory / name).write_bytes(png(name))
            metadata = dict(operation='PL', outputs=['plot_linear.dat', 'plot_linear.png', 'plot_log.png'],
                figures={name: {'data_file': 'plot_linear.dat', 'plot': {'scale': role}}
                         for name, role in [('plot_linear.png', 'linear'), ('plot_log.png', 'log')]},
                pl_display={'version': 1})
            (directory / 'plot_linear.metadata.json').write_text(json.dumps(metadata))
            index = FigureIndex(root)
            records = index.scan()
            self.assertEqual({r.role for r in records}, {'linear', 'log'})
            self.assertEqual(len(index.match_image(png('different metadata'))), 2)
            self.assertEqual(index.get(records[0].record_id), records[0])
            self.assertEqual(records[0].snapshot, {'version': 1})
            (directory / 'plot_linear.dat').unlink()
            self.assertEqual(index.scan(), [])

    def test_real_pl_and_drr_exports(self):
        from dataclasses import replace
        import numpy as np
        from core.loader import DataCube
        from core.plotting import HeatmapParams
        from core.export import export_pl_pngs_and_dat, export_drr_png_and_dat
        cube = DataCube(np.array([1., 2., 3.]), np.array([0., 1.]),
                        np.array([[1., 2., 3.], [2., 3., 4.]]), 'Gate', 'Sample', 'Signal')
        params = HeatmapParams(title='Sample', xlabel='Energy', ylabel='Gate',
                               cbar_label='Signal', vmin=1., vmax=4., xlim=(1., 3.), ylim=(0., 1.), png_width_scale=2.)
        with TemporaryDirectory() as tmp:
            pl = export_pl_pngs_and_dat(tmp, 'sample.csv', cube_linear=cube, cube_log=cube,
                params_linear=params, params_log=replace(params, title='Log'),
                include_original_png=True, processed_name='Processed Data/PL',
                metadata_extra={'pl_display': {'version': 1}})
            raw = export_drr_png_and_dat(tmp, cube=cube, params=params, export_base='sample_raw',
                processed_name='Processed Data/DRR', include_original_png=True,
                metadata_processing={'derivative_order': None}, metadata_extra={'ui_state': {'page': 'raw'}})
            second = export_drr_png_and_dat(tmp, cube=cube, params=replace(params, png_width_scale=1.),
                export_base='sample_second', processed_name='Processed Data/DRR',
                metadata_processing={'derivative_order': 2})
            index = FigureIndex(tmp)
            records = index.scan()
            lookup = {r.png_path: r for r in records}
            for key, role in [('png_linear', 'linear'), ('png_log', 'log'),
                              ('png_linear_wide', 'linear_wide'), ('png_log_wide', 'log_wide')]:
                self.assertEqual(lookup[pl[key].resolve()].role, role)
                self.assertEqual(lookup[pl[key].resolve()].dat_path, pl['dat'].resolve())
            self.assertEqual(lookup[raw['png'].resolve()].role, 'raw')
            self.assertEqual(lookup[raw['png_wide'].resolve()].role, 'raw_wide')
            self.assertEqual(lookup[second['png'].resolve()].role, 'second')
            self.assertEqual(lookup[raw['png'].resolve()].snapshot, {'page': 'raw'})
            self.assertEqual(len(index.register_exports(pl, 'pl', {})), 4)

    def test_legacy_pl_and_malformed_metadata(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'Processed Data' / 'PL'
            folder.mkdir(parents=True)
            (folder / 'sample.dat').write_text('data')
            (folder / 'sample_linear.png').write_bytes(png())
            (folder / 'sample_log.png').write_bytes(png())
            payload = {'operation': 'PL', 'outputs': ['sample.dat', 'sample_linear.png', 'sample_log.png'],
                       'plot': {'linear': {'title': 'A'}, 'log': {'title': 'B'}}}
            meta = folder / 'sample.metadata.json'
            meta.write_text(json.dumps(payload))
            index = FigureIndex(tmp)
            self.assertEqual({r.role: r.plot['title'] for r in index.scan()}, {'linear': 'A', 'log': 'B'})
            payload['outputs'].append('other.dat')
            meta.write_text(json.dumps(payload))
            self.assertEqual(index.scan(), [])

    def test_explicit_role_overrides_filename_and_rejects_invalid(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'Processed Data' / 'PL'
            folder.mkdir(parents=True)
            (folder / 'new.dat').write_text('data')
            (folder / 'new.png').write_bytes(png())
            metadata = {'operation': 'PL', 'outputs': ['new.dat', 'new.png'],
                'figures': {'new.png': {'data_file': 'new.dat', 'plot': {'title': 'A'}, 'role': 'log'}}}
            path = folder / 'new.metadata.json'
            path.write_text(json.dumps(metadata))
            index = FigureIndex(tmp)
            self.assertEqual(index.scan()[0].role, 'log')
            metadata['figures']['new.png']['role'] = 'second'
            path.write_text(json.dumps(metadata))
            self.assertEqual(index.scan(), [])

    def test_targeted_lookup_and_selective_decode(self):
        from unittest.mock import patch
        import core.pptx_figure_index as module
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'Processed Data' / 'PL'
            folder.mkdir(parents=True)
            (folder / 'plot.dat').write_text('data')
            names = ['first_linear.png', 'second_linear.png', 'other_linear.png']
            for name in names:
                (folder / name).write_bytes(png(name))
            meta = {'operation': 'PL', 'outputs': ['plot.dat', *names],
                    'figures': {name: {'data_file': 'plot.dat', 'plot': {}} for name in names}}
            (folder / 'plot.metadata.json').write_text(json.dumps(meta))
            index = FigureIndex(tmp)
            original = module._fingerprint
            with patch.object(module, '_fingerprint', wraps=original) as decode:
                records = index.find_png(folder / names[0])
                self.assertEqual(len(records), 1)
                self.assertEqual(decode.call_count, 1)
                self.assertEqual(index.get(records[0].record_id), records[0])
            self.assertEqual(len(index.lookup_image(png('different text'))), 3)
            # Recompression is pixel-identical but has a different IDAT stream.
            stream = BytesIO()
            Image.new('RGBA', (2, 3), (12, 34, 56, 255)).save(stream, format='PNG', compress_level=0)
            self.assertEqual(len(index.lookup_image(stream.getvalue())), 3)
            # Changed file invalidates cached structural and pixel signatures.
            Image.new('RGBA', (2, 3), (1, 2, 3, 255)).save(folder / names[0])
            self.assertEqual(len(index.lookup_image(png())), 2)

    def test_duplicate_sidecars_have_scan_dedup_semantics(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'Processed Data' / 'PL'
            folder.mkdir(parents=True)
            image = folder / 'sample_linear.png'
            image.write_bytes(png())
            (folder / 'sample.dat').write_text('data')
            payload = {'operation': 'PL', 'outputs': ['sample.dat', image.name],
                       'figures': {image.name: {'data_file': 'sample.dat', 'plot': {'title': 'first'}}}}
            (folder / 'a.metadata.json').write_text(json.dumps(payload))
            payload['figures'][image.name]['plot']['title'] = 'second'
            (folder / 'b.metadata.json').write_text(json.dumps(payload))
            index = FigureIndex(tmp)
            expected = index.scan()
            self.assertEqual(len(expected), 1)
            for result in (index.find_png(image), index.lookup_image(png()),
                           index.register_exports({'png_linear': image}, 'pl', {})):
                self.assertEqual(result, expected)

    def test_size_prefilter_and_tag_revalidate(self):
        from unittest.mock import patch
        import core.pptx_figure_index as module
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'Processed Data' / 'PL'
            folder.mkdir(parents=True)
            (folder / 'data.dat').write_text('data')
            target = folder / 'a_linear.png'
            target.write_bytes(png())
            other = folder / 'b_linear.png'
            Image.new('RGBA', (4, 5), 'red').save(other)
            payload = {'operation': 'PL', 'outputs': ['data.dat', target.name, other.name],
                       'figures': {name: {'data_file': 'data.dat', 'plot': {'title': 'old'}}
                                   for name in (target.name, other.name)}}
            sidecar = folder / 'data.metadata.json'
            sidecar.write_text(json.dumps(payload))
            index = FigureIndex(tmp)
            # Recompression guarantees the structural prefilter misses.
            stream = BytesIO()
            Image.new('RGBA', (2, 3), (12, 34, 56, 255)).save(stream, format='PNG', compress_level=0)
            original = module._fingerprint
            with patch.object(module, '_fingerprint', wraps=original) as decode:
                records = index.lookup_image(stream.getvalue())
                self.assertEqual(len(records), 1)
                self.assertEqual(decode.call_count, 2)  # input and one same-size candidate
            payload['figures'][target.name]['plot']['title'] = 'updated'
            sidecar.write_text(json.dumps(payload))
            verified = index.lookup_image(stream.getvalue(), records[0].record_id)
            self.assertEqual(verified[0].plot['title'], 'updated')
            Image.new('RGBA', (2, 3), 'blue').save(target)
            self.assertEqual(index.lookup_image(stream.getvalue(), records[0].record_id), [])
            payload['figures'][target.name]['plot']['title'] = 'pending'
            payload['display_save_pending'] = True
            sidecar.write_text(json.dumps(payload))
            self.assertEqual(index.find_png(target), [])

    def test_find_png_reads_only_sibling_sidecars_and_target_files(self):
        from unittest.mock import patch
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'Processed Data'
            folder = root / 'PL'
            folder.mkdir(parents=True)
            elsewhere = root / 'DRR'
            elsewhere.mkdir()
            target, unrelated = folder / 'a_linear.png', folder / 'b_linear.png'
            target.write_bytes(png())
            unrelated.write_bytes(png())
            (folder / 'data.dat').write_text('data')
            payload = {'operation': 'PL', 'outputs': ['data.dat', target.name, unrelated.name],
                       'figures': {name: {'data_file': 'data.dat', 'plot': {}}
                                   for name in (target.name, unrelated.name)}}
            (folder / 'data.metadata.json').write_text(json.dumps(payload))
            (elsewhere / 'other.metadata.json').write_text('{}')
            index = FigureIndex(tmp)
            original = Path.is_file
            def checked(path):
                if path.name == unrelated.name:
                    raise AssertionError('Unrelated PNG was inspected')
                return original(path)
            with patch.object(Path, 'is_file', checked), patch.object(index, '_metadata', wraps=index._metadata) as read:
                self.assertEqual(len(index.find_png(target)), 1)
                self.assertEqual(read.call_count, 1)
            self.assertEqual(index.find_png(elsewhere.parent.parent / 'outside.png'), [])

    def test_chunk_length_prefilter_handles_metadata_split_and_recompression(self):
        import struct
        import zlib
        from unittest.mock import patch
        from core.pptx_figure_index import _png_chunk_lengths, _png_content_fingerprint
        original = png('original metadata')
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        offset=8; split=original[:8]
        while offset < len(original):
            size=struct.unpack_from('>I',original,offset)[0]
            kind=original[offset+4:offset+8]; data=original[offset+8:offset+8+size]
            if kind == b'IDAT':
                middle=len(data)//2
                split += chunk(kind,data[:middle])+chunk(kind,data[middle:])
            else:
                split += original[offset:offset+size+12]
            offset += size+12
        self.assertEqual(_png_content_fingerprint(original), _png_content_fingerprint(split))
        self.assertEqual(_png_chunk_lengths(BytesIO(original),len(original)), _png_chunk_lengths(BytesIO(split),len(split)))
        self.assertEqual(_png_chunk_lengths(BytesIO(original),len(original)), _png_chunk_lengths(BytesIO(png('different metadata')),len(png('different metadata'))))
        self.assertIsNone(_png_chunk_lengths(BytesIO(original[:-3]),len(original)-3))
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)/'Processed Data'/'PL';folder.mkdir(parents=True)
            (folder/'data.dat').write_text('data')
            (folder/'test_linear.png').write_bytes(original)
            (folder/'test.metadata.json').write_text(json.dumps({'operation':'PL','outputs':['data.dat','test_linear.png'],
                'figures':{'test_linear.png':{'data_file':'data.dat','plot':{}}}}))
            index=FigureIndex(tmp)
            self.assertEqual(len(index.lookup_image(split)),1)
            stream=BytesIO();Image.new('RGBA',(2,3),(12,34,56,255)).save(stream,format='PNG',compress_level=0)
            self.assertEqual(len(index.lookup_image(stream.getvalue())),1)
            with patch('core.pptx_figure_index._fingerprint', side_effect=AssertionError('input decoded')):
                # Image and selected file hashes already cached; supplied pixels avoids re-decoding.
                digest=index._hash(folder/'test_linear.png')
                self.assertEqual(len(index.lookup_image(split,target_pixels=digest)),1)

    def test_pptx_embedded_picture_and_external_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'deck.pptx'
            p = 'http://schemas.openxmlformats.org/presentationml/2006/main'
            r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
            rel = 'http://schemas.openxmlformats.org/package/2006/relationships'
            def make(target, mode=''):
                with ZipFile(path, 'w') as z:
                    z.writestr('ppt/presentation.xml', f'<p:presentation xmlns:p="{p}" xmlns:r="{r}"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>')
                    z.writestr('ppt/_rels/presentation.xml.rels', f'<Relationships xmlns="{rel}"><Relationship Id="rId1" Target="slides/slide1.xml"/></Relationships>')
                    z.writestr('ppt/slides/slide1.xml', f'<p:sld xmlns:p="{p}" xmlns:r="{r}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:grpSp><p:pic><p:nvPicPr><p:cNvPr id="42"/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/></p:blipFill></p:pic></p:grpSp></p:sld>')
                    z.writestr('ppt/slides/_rels/slide1.xml.rels', f'<Relationships xmlns="{rel}"><Relationship Id="rId2" Target="{target}" {mode}/></Relationships>')
                    z.writestr('ppt/media/image1.png', png())
            make('../media/image1.png')
            self.assertEqual(read_shape_image(path, 256, 42), png())
            make('https://example.com/image.png', 'TargetMode="External"')
            with self.assertRaises(ValueError):
                read_shape_image(path, 256, 42)
            make('../../../outside.png')
            with self.assertRaises(ValueError):
                read_shape_image(path, 256, 42)

if __name__ == '__main__':
    unittest.main()
