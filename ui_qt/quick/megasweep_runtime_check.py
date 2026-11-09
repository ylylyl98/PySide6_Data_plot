"""Isolated numerical/FFmpeg smoke for source and frozen Megasweep builds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    report = {'status': 'error', 'frozen': bool(getattr(sys, 'frozen', False))}
    try:
        import numpy as np
        import pandas as pd
        import imageio_ffmpeg
        from core.megasweep.workflow import load_dataset, analyze, export_result, export_video, MoviePreviewSession
        executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
        assert executable.is_file(), 'Bundled FFmpeg executable missing'
        if report['frozen']:
            assert executable.is_relative_to(Path(sys._MEIPASS)), 'FFmpeg was not bundled in the application'
        with tempfile.TemporaryDirectory(prefix='dptk-megasweep-check-') as temporary:
            root = Path(temporary)
            wavelengths = np.linspace(800., 1000., 51)
            columns = ['Vbg', 'Vtg', *map(str, wavelengths)]
            rows = [[x, y, *(500 + (50+x*5+y)*np.exp(-((wavelengths-900)/20)**2))]
                    for x in (-1.,0.,1.) for y in (-1.,0.,1.)]
            optical = root / 'optical.csv'
            pd.DataFrame(rows, columns=columns).to_csv(optical, index=False)
            background = root / 'background.csv'
            pd.DataFrame([[0,0,*np.full(51,500.)]], columns=columns).to_csv(background,index=False)
            transport = root / 'transport.csv'
            pd.DataFrame([(x,y,(x+2)*(y+2)) for x in (-1,0,1) for y in (-1,0,1)],
                         columns=['Vbg','Vtg','Current']).to_csv(transport,index=False)
            parameters = dict(view='intensity', auto_baseline=False, baseline=0., energy_min=1.25, energy_max=1.5,
                              cut_type='x', cut_value=0., auto_background_scale=False,
                              movie_options=dict(width=320,height=240,fps=5,hold=.2,head=0.,tail=0.,repeats=1))
            outputs = []
            data = load_dataset(optical)
            for mode, path, values, references in (
                ('PL', optical, parameters, ()),
                ('Reflection', optical, dict(parameters,view='rc_p2p'), (str(background),)),
                ('Transport', transport, dict(view='transport_map',x_column='Vbg',y_column='Vtg',channel='Current'), ())):
                result = analyze(load_dataset(path, mode), values, references)
                outputs.extend(export_result(result, root / mode, parameters=values))
            preview = MoviePreviewSession(data, parameters)
            try:
                assert preview.frame(0)['png'].startswith(b'\x89PNG')
            finally:
                preview.close()
            # Exercise the bundled executable even if the developer has system FFmpeg.
            original_path = os.environ.get('PATH')
            os.environ['PATH'] = ''
            try:
                movie = export_video(data, parameters, (), root / 'movies')
            finally:
                if original_path is None:
                    os.environ.pop('PATH', None)
                else:
                    os.environ['PATH'] = original_path
            assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in [*outputs,*movie])
            metadata = json.loads(Path(movie[1]).read_text(encoding='utf-8'))
            report.update(status='ok', modes=['PL','Reflection','Transport'], export_files=len(outputs),
                          ffmpeg_bundled=report['frozen'], ffmpeg=str(executable), movie=metadata,
                          widgets_imported='PySide6.QtWidgets' in sys.modules)
    except Exception as exc:
        report['error'] = str(exc) or type(exc).__name__
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    if sys.stdout is not None:
        print(json.dumps(report, indent=2, default=str))
    return 0 if report['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
