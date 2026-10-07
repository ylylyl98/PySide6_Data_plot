"""Real detached-workspace lifecycle: handoff, graceful close, fresh restore."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from core.peak_workspace import create_dataset, save_workspace, load_workspace
from tests.test_peak_workspace import sample


class ProcessTests(unittest.TestCase):
    def test_single_instance_handoff_and_fresh_process_restore(self):
        root=Path(__file__).resolve().parents[1]
        code='''
import json, sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import run_peak_analysis
from core.pl_peak_analysis import detect
original=QApplication.exec
def run(app):
    timer=QTimer()
    def observe():
        for window in app.topLevelWidgets():
            if isinstance(window,run_peak_analysis.PeakAnalysisWindow) and window.datasets and not window.busy:
                for d in window.datasets.values():
                    if not d['branches']:
                        d['branches']=detect(d,4,d['settings'])
                        d['branches'][0]['name']='Saved branch'
                Path(sys.argv[2]).write_text(json.dumps({'count':len(window.datasets)}))
                if Path(sys.argv[1]).exists():window.close()
    timer.timeout.connect(observe);timer.start(100)
    return original()
QApplication.exec=run
raise SystemExit(run_peak_analysis.main(sys.argv[3:]))
'''
        env=dict(os.environ,QT_QPA_PLATFORM='offscreen')
        flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);session=folder/'workspace.npz';stop=folder/'stop';ready=folder/'ready.json'
            paths=[folder/'pl.npz',folder/'drr.npz']
            for path,kind in zip(paths,['PL','DRR']):
                d=create_dataset(sample(kind),kind,'second' if kind=='DRR' else 'raw',kind+'.csv',kind)
                d['settings']['detection_method']='sg'  # Simulate an older main-app handoff.
                save_workspace(path,[d],d['key'])
            process=subprocess.Popen([sys.executable,'-c',code,str(stop),str(ready),'--session',str(session),'--snapshot',str(paths[0])],
                cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=flags)
            def wait_count(count):
                deadline=time.monotonic()+30
                while time.monotonic()<deadline:
                    if process.poll() is not None:self.fail(process.communicate()[1].decode(errors='replace'))
                    try:
                        if json.loads(ready.read_text())['count']==count:return
                    except (OSError,ValueError):pass
                    time.sleep(.1)
                self.fail('Analysis process did not accept the snapshot.')
            try:
                wait_count(1)
                forwarded=subprocess.run([sys.executable,str(root/'run_peak_analysis.py'),'--session',str(session),'--snapshot',str(paths[1])],
                    cwd=root,env=env,capture_output=True,timeout=30,creationflags=flags)
                self.assertEqual(forwarded.returncode,0,forwarded.stderr.decode(errors='replace'))
                wait_count(2);stop.touch();_,errors=process.communicate(timeout=25)
                self.assertEqual(process.returncode,0,errors.decode(errors='replace'))
                datasets,_=load_workspace(session);self.assertEqual(len(datasets),2)
                self.assertTrue(all(d['branches'][0]['name']=='Saved branch' for d in datasets))
                derivative=next(d for d in datasets if d['kind']=='DRR')
                self.assertEqual(derivative['settings']['detection_method'],'local')
                derivative['settings']['detection_method']='sg'  # Saved manual override.
                save_workspace(session,datasets,derivative['key'])
                fresh=subprocess.run([sys.executable,'-c',code,str(stop),str(ready),'--session',str(session)],
                    cwd=root,env=env,capture_output=True,timeout=30,creationflags=flags)
                self.assertEqual(fresh.returncode,0,fresh.stderr.decode(errors='replace'))
                self.assertEqual(len(load_workspace(session)[0]),2)
                restored=next(d for d in load_workspace(session)[0] if d['kind']=='DRR')
                self.assertEqual(restored['settings']['detection_method'],'sg')
            finally:
                if process.poll() is None:process.kill();process.communicate()


if __name__=='__main__':unittest.main()
