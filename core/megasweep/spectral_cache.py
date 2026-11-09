"""Bounded numeric and temporary image caches for spectral rendering jobs."""
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


class SpectralSliceCache:
    def __init__(self, cut_factory, cut_budget=64 * 1024 * 1024, frame_budget=512 * 1024 * 1024):
        self.cut_factory = cut_factory
        self.cut_budget = cut_budget
        self.cut_bytes = 0
        self.cuts = OrderedDict()
        self.statistics = {}
        self.frame_directory = TemporaryDirectory(prefix='megasweep-frames-')
        self.frame_budget = frame_budget
        self.frame_bytes = 0
        self.frame_cache = OrderedDict()
        self.frame_index = 0

    def get_cut(self, value):
        if value in self.cuts:
            self.cuts.move_to_end(value)
            return self.cuts[value][0]
        cut = self.cut_factory(value)
        size = sum(array.nbytes for array in cut.values() if isinstance(array, np.ndarray))
        if size <= self.cut_budget:
            # Own arrays so small views cannot retain an entire source matrix.
            cut = {name: array.copy() if isinstance(array, np.ndarray) else array
                   for name, array in cut.items()}
            while self.cuts and self.cut_bytes + size > self.cut_budget:
                _, (_, previous_size) = self.cuts.popitem(last=False)
                self.cut_bytes -= previous_size
            self.cuts[value] = (cut, size)
            self.cut_bytes += size
        return cut

    def cached_frame(self, key, render):
        if key in self.frame_cache:
            path, _ = self.frame_cache[key]
            with Image.open(path) as cached:
                image = np.array(cached.convert('RGB'))
            self.frame_cache.move_to_end(key)
            return image
        image = render()
        if self.frame_budget <= 0:
            return image
        path = Path(self.frame_directory.name) / f'{self.frame_index}.png'
        self.frame_index += 1
        Image.fromarray(image).save(path, compress_level=1)
        size = path.stat().st_size
        if size <= self.frame_budget:
            while self.frame_cache and self.frame_bytes + size > self.frame_budget:
                _, (previous_path, previous_size) = self.frame_cache.popitem(last=False)
                previous_path.unlink()
                self.frame_bytes -= previous_size
            self.frame_cache[key] = (path, size)
            self.frame_bytes += size
        else:
            path.unlink()
        return image

    def close(self):
        self.frame_directory.cleanup()
        self.frame_cache.clear()
        self.frame_bytes = 0
        self.cuts.clear()
        self.cut_bytes = 0
        self.statistics.clear()
