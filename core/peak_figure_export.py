"""Save matching original/wide source overlays without overwriting earlier files."""
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

from core.peak_source_figure import render_source_figure, source_export_width
from core.typography import save_figure


def export_source_figures(destination, dataset, *, cancelled=None):
    from matplotlib import pyplot as plt
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width = source_export_width(dataset)
    variants = [(destination.name, 1.)]
    if width > 1.:
        variants.append((f'{destination.stem}_{width}x{destination.suffix}', width))

    def check():
        if cancelled and cancelled():
            raise InterruptedError('Peak Analysis export cancelled.')

    with TemporaryDirectory(prefix='.peak-figure-', dir=destination.parent) as temporary:
        staged = Path(temporary)
        for name, scale in variants:
            check()
            figure = render_source_figure(dataset, width_scale=scale)
            try:
                save_figure(figure, staged / name, dpi=figure.dpi,
                            facecolor=figure.get_facecolor(), edgecolor='none',
                            bbox_inches=figure.bbox_inches, pad_inches=0)
            finally:
                plt.close(figure)
        number = 1
        while True:
            check()
            suffix = '' if number == 1 else f'_{number:02d}'
            targets = [destination.parent / (Path(name).stem + suffix + Path(name).suffix)
                       for name, _ in variants]
            if any(target.exists() for target in targets):
                number += 1
                continue
            created = []
            try:
                for (name, _), target in zip(variants, targets):
                    check()
                    with target.open('xb') as output:
                        created.append(target)
                        with (staged / name).open('rb') as source:
                            copyfileobj(source, output)
                check()
                return [str(target) for target in targets]
            except FileExistsError:
                for target in created:
                    target.unlink(missing_ok=True)
                number += 1
            except BaseException:
                for target in created:
                    target.unlink(missing_ok=True)
                raise
