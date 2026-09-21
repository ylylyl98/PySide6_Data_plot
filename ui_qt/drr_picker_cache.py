"""UI-owned freshness leases; filesystem changes invalidate in-flight scans."""
import os
import time


class PickerFreshness:
    MAX_AGE = 30.0

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.epoch = 0
        self.checked = {}

    def key(self, folder, include_all):
        return os.path.normcase(os.path.abspath(folder)), bool(include_all)

    def begin(self, folder, include_all):
        return self.key(folder, include_all), self.epoch

    def complete(self, token):
        key, epoch = token
        if epoch == self.epoch:
            self.checked[key] = self.clock()
            while len(self.checked) > 4:
                self.checked.pop(next(iter(self.checked)))

    def invalidate(self):
        self.epoch += 1
        self.checked.clear()

    def due(self, folder, include_all):
        stamp = self.checked.get(self.key(folder, include_all))
        return stamp is None or self.clock() - stamp >= self.MAX_AGE


def freshness(owner):
    if not hasattr(owner, '_drr_picker_freshness'):
        owner._drr_picker_freshness = PickerFreshness()
    return owner._drr_picker_freshness
