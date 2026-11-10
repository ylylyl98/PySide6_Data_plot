import json
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import numpy as np
from core.loader import DataCube
from core.peak_workspace import create_dataset, load_workspace, save_workspace


def check_workspace_metadata_encoding_and_legacy_compatibility(tmp_path):
    cube = DataCube(np.array([1., 2., 3.]), np.array([0., 1.]), np.arange(6.).reshape(2, 3), 'Gate', 'μ test', 'Intensity')
    dataset = create_dataset(cube, 'DRR', 'raw', 'source', 'μ test')
    dataset['custom'] = {'label': 'ΔR/R', 'values': [1, 2, None]}
    path = tmp_path / 'workspace.npz'
    save_workspace(path, [dataset], dataset['key'])
    with ZipFile(path) as archive:
        assert archive.getinfo('metadata.npy').compress_type == ZIP_DEFLATED
        assert archive.getinfo('0_Z.npy').compress_type == ZIP_STORED
    with np.load(path, allow_pickle=False) as archive:
        assert archive['metadata'].dtype == np.uint8
        arrays = {key: archive[key] for key in archive.files}
    restored, active = load_workspace(path)
    assert active == dataset['key']
    assert restored[0]['custom'] == dataset['custom']
    np.testing.assert_array_equal(restored[0]['cube'].Z, cube.Z)
    metadata = arrays['metadata'].tobytes().decode('utf-8')
    assert json.loads(metadata)['schema'] == 2
    legacy_metadata = json.loads(metadata)
    legacy_metadata['schema'] = 1
    metadata = json.dumps(legacy_metadata)
    arrays['metadata'] = np.array(metadata)
    legacy = tmp_path / 'legacy.npz'
    np.savez(legacy, **arrays)
    old, old_active = load_workspace(legacy)
    assert old_active == active
    assert old[0]['custom'] == dataset['custom']
    np.testing.assert_array_equal(old[0]['cube'].Z, cube.Z)


import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


class Tests(unittest.TestCase):
    def test_workspace_metadata_encoding_and_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            check_workspace_metadata_encoding_and_legacy_compatibility(Path(directory))
