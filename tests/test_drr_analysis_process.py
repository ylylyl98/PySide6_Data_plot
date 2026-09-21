"""Exercise real local-socket handoff and graceful standalone shutdown."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np
from core.loader import DataCube
from core.drr_analysis_workspace import create_dataset
from core.drr_workspace_session import save_session, load_session


class ProcessTests(unittest.TestCase):
    def test_second_launch_imports_into_existing_process_and_reopens(self):
        root=Path(__file__).resolve().parents[1]
        code="""
import sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import run_drr_analysis
original=QApplication.exec
def run(app):
    timer=QTimer()
    timer.timeout.connect(lambda: app.closeAllWindows() if Path(sys.argv[1]).exists() else None)
    timer.start(100)
    return original()
QApplication.exec=run
raise SystemExit(run_drr_analysis.main(sys.argv[2:]))
"""
        env=dict(os.environ,QT_QPA_PLATFORM='offscreen')
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);session=folder/'workspace.npz';stop=folder/'stop'
            x=np.linspace(1,1.1,101);y=np.arange(3.)
            datasets=[create_dataset(DataCube(x,y,np.ones((3,101))*i,'Y',str(i),'DR/R'),str(i),{}) for i in (1,2)]
            snapshots=[folder/f'{i}.npz' for i in (1,2)]
            for path,d in zip(snapshots,datasets):save_session(path,[d])
            def wait_count(count):
                deadline=time.monotonic()+25
                while time.monotonic()<deadline:
                    if process.poll() is not None:
                        output=process.communicate()
                        self.fail(output[1].decode(errors='replace'))
                    if session.exists() and len(load_session(session)[0])==count:return
                    time.sleep(.1)
                self.fail('Standalone workspace did not persist the expected datasets')
            process=subprocess.Popen([sys.executable,'-c',code,str(stop),str(folder),'--session',str(session),'--snapshot',str(snapshots[0])],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                wait_count(1)
                second=subprocess.run([sys.executable,str(root/'run_drr_analysis.py'),str(folder),'--session',str(session),'--snapshot',str(snapshots[1])],cwd=root,env=env,capture_output=True,timeout=25)
                self.assertEqual(second.returncode,0,second.stderr.decode(errors='replace'))
                wait_count(2)
                stop.touch();output=process.communicate(timeout=20)
                self.assertEqual(process.returncode,0,output[1].decode(errors='replace'))
                self.assertEqual({d.key for d in load_session(session)[0]},{d.key for d in datasets})
                reopened=subprocess.run([sys.executable,'-c',code,str(stop),str(folder),'--session',str(session)],cwd=root,env=env,capture_output=True,timeout=25)
                self.assertEqual(reopened.returncode,0,reopened.stderr.decode(errors='replace'))
                self.assertEqual(len(load_session(session)[0]),2)
            finally:
                if process.poll() is None:process.kill();process.communicate()
