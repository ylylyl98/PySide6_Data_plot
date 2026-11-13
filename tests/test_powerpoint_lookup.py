from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from PIL import Image
from ui_qt.quick.powerpoint_lookup import PictureLookup


def image(color='red'):
    data = BytesIO(); Image.new('RGBA', (3, 4), color).save(data, format='PNG'); return data.getvalue()


class LookupTests(TestCase):
    def test_preparation_avoids_decoding_and_respects_cancellation(self):
        with TemporaryDirectory() as tmp:
            root,png,meta,payload=self.setup_files(tmp)
            lookup=PictureLookup(root,Path(tmp)/'cache')
            with mock.patch.object(lookup.index,'_hash',side_effect=AssertionError('unexpected decode')), mock.patch.object(lookup.index,'_structural_hash',side_effect=AssertionError('unexpected full read')):
                lookup.prepare(lambda: True)
                self.assertFalse(lookup.index._size_cache)
                lookup.prepare()
                self.assertTrue(lookup.index._size_cache)
                self.assertTrue(lookup.index._length_cache)
            png.write_bytes(image('blue'))
            self.assertEqual(len(lookup.match(image('blue'))[0]),1)
            self.assertEqual(lookup.match(image())[0],[])

    def setup_files(self, tmp):
        root = Path(tmp)/'experiment'; folder = root/'Processed Data'/'PL'; folder.mkdir(parents=True)
        png = folder/'test_linear.png'; png.write_bytes(image())
        (folder/'data.dat').write_text('data')
        meta = folder/'data.metadata.json'
        payload = {'operation':'PL','outputs':['data.dat',png.name], 'figures':{png.name:{'data_file':'data.dat','plot':{'title':'old'}}}}
        meta.write_text(json.dumps(payload))
        return root,png,meta,payload

    def test_restart_hint_validates_pixels_and_metadata_without_full_lookup(self):
        with TemporaryDirectory() as tmp:
            root,png,meta,payload = self.setup_files(tmp); cache=Path(tmp)/'cache'
            first = PictureLookup(root,cache); record=first.find_png(png)[0]
            second = PictureLookup(root,cache)
            payload['figures'][png.name]['plot']['title']='updated';meta.write_text(json.dumps(payload))
            with mock.patch.object(second.index,'lookup_image',side_effect=AssertionError('unexpected full scan')):
                result,digest = second.match(image(),record.record_id)
            self.assertEqual(result[0].plot['title'],'updated'); self.assertEqual(len(digest),64)
            png.write_bytes(image('blue'))
            self.assertEqual(second.match(image(),record.record_id)[0],[])
            payload['display_save_pending']=True; meta.write_text(json.dumps(payload))
            self.assertEqual(second.find_png(png),[])

    def test_outside_hint_rejected_and_malformed_cache_ignored(self):
        with TemporaryDirectory() as tmp:
            root,png,meta,payload=self.setup_files(tmp); cache=Path(tmp)/'cache'
            first=PictureLookup(root,cache); record=first.find_png(png)[0]
            first.cache_path.write_text(json.dumps({'version':1,'records':{record.record_id:'../outside.png'}}))
            second=PictureLookup(root,cache)
            self.assertEqual(second._hints,{})
            first.cache_path.write_text('{bad json')
            self.assertEqual(PictureLookup(root,cache)._hints,{})

    def test_untagged_duplicate_pixels_preserved(self):
        with TemporaryDirectory() as tmp:
            root,png,meta,payload=self.setup_files(tmp)
            other=png.with_name('other_linear.png');other.write_bytes(image())
            payload['outputs'].append(other.name)
            payload['figures'][other.name]={'data_file':'data.dat','plot':{}}
            meta.write_text(json.dumps(payload))
            records,_=PictureLookup(root,Path(tmp)/'cache').match(image())
            self.assertEqual({r.png_path.name for r in records},{png.name,other.name})
