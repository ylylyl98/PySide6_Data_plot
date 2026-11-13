"""Read current PowerPoint images using a verified saved-file baseline."""
from collections import OrderedDict
from pathlib import Path

from core.pptx_figure_index import read_shape_image
from ui_qt.quick.powerpoint_temp import temporary_directory


class PresentationImageReader:
    def __init__(self):
        # Hold the COM proxy itself: object identity must not be inferred from a
        # reused integer id or a same-named presentation in another instance.
        self._baselines = OrderedDict()
        self._verified_images = OrderedDict()

    def clear(self):
        self._baselines.clear()
        self._verified_images.clear()

    def remember_verified(self, pres, slide_id, shape_id, image):
        """Remember bytes only after live document verification has succeeded."""
        key = (int(slide_id), int(shape_id))
        self._verified_images[key] = (pres, image)
        self._verified_images.move_to_end(key)
        while len(self._verified_images) > 16:
            self._verified_images.popitem(last=False)

    @staticmethod
    def matches_live_image(pres, slide_id, shape_id, expected):
        """An exported candidate is trusted only when byte-identical to known PNG."""
        if not expected or not expected.startswith(b'\x89PNG\r\n\x1a\n'):
            return False
        try:
            from ui_qt.quick.powerpoint_emf import export_shape_png
            slide = pres.Slides.FindBySlideID(int(slide_id))
            for index in range(1, slide.Shapes.Count + 1):
                shape = slide.Shapes(index)
                if int(shape.Id) == int(shape_id):
                    return export_shape_png(shape) == expected
        except Exception:
            # Unsupported exports, effects and Office versions use the full
            # document path. Never treat an export error as image equivalence.
            pass
        return False

    @staticmethod
    def _saved_source(pres):
        if int(pres.Saved) != -1:
            return None
        name = str(pres.FullName)
        path = Path(name)
        if path.suffix.lower() != '.pptx' or not path.is_file():
            return None
        st = path.stat()
        identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        return name, path, identity

    def read(self, pres, slide_id, shape_id):
        key = (int(slide_id), int(shape_id))
        try:
            before = self._saved_source(pres)
        except Exception:
            before = None
        baseline = self._baselines.get(key)
        if before is not None and baseline is not None:
            owner, name, identity, expected = baseline
            if owner is pres and before[0] == name and before[2] == identity:
                try:
                    disk = read_shape_image(before[1], *key)
                    after = self._saved_source(pres)
                    if after is not None and after[0] == name and after[2] == identity and disk == expected:
                        self._baselines.move_to_end(key)
                        return disk
                except Exception:
                    pass
        self._baselines.pop(key, None)
        verified = self._verified_images.get(key)
        if verified is not None:
            if self.matches_live_image(pres, *key, verified[1]):
                self._verified_images.move_to_end(key)
                return verified[1]
        # A saved file can supply a candidate even when the live deck is
        # dirty or uses another COM proxy. Never trust it without comparing
        # against this exact live slide and shape; a stale file simply misses.
        try:
            candidate_path = Path(str(pres.FullName))
            if candidate_path.suffix.lower() == '.pptx' and candidate_path.is_file():
                candidate = read_shape_image(candidate_path, *key)
                if self.matches_live_image(pres, *key, candidate):
                    self.remember_verified(pres, *key, candidate)
                    return candidate
        except Exception:
            pass
        with temporary_directory() as directory:
            snapshot = Path(directory) / 'snapshot.pptx'
            pres.SaveCopyAs(str(snapshot), 24)
            actual = read_shape_image(snapshot, *key)
        self.remember_verified(pres, *key, actual)
        # Failure to validate an optional baseline must never affect the fresh
        # snapshot result. A dirty deck cannot establish or reuse a baseline.
        if before is not None:
            try:
                middle = self._saved_source(pres)
                if middle is not None and middle[0] == before[0] and middle[2] == before[2]:
                    disk = read_shape_image(before[1], *key)
                    after = self._saved_source(pres)
                    if after is not None and after[0] == before[0] and after[2] == before[2] and disk == actual:
                        self._baselines[key] = (pres, before[0], before[2], actual)
                        self._baselines.move_to_end(key)
                        while len(self._baselines) > 16:
                            self._baselines.popitem(last=False)
            except Exception:
                pass
        return actual
