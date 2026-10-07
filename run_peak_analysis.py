"""Fresh-process entry point for the independently restartable analysis UI."""
import argparse
import hashlib
from pathlib import Path
import sys

from PySide6.QtCore import QTimer, QProcess
from PySide6.QtWidgets import QApplication

from core.peak_workspace import session_directory, load_workspace
from ui_qt.peak_analysis_ipc import send_message, MessageServer
from ui_qt.peak_analysis_window import PeakAnalysisWindow


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--session',type=Path,default=session_directory()/'workspace.npz')
    parser.add_argument('--reply-server',default='')
    args=parser.parse_args(argv)
    app=QApplication([sys.argv[0]]);app.setApplicationName('DPTK Peak Analysis');app.setOrganizationName('ylylyl98');app.setStyle('Fusion')
    name='DPTK-Peak-'+hashlib.sha256(str(args.session.resolve()).casefold().encode()).hexdigest()[:24]
    message={'snapshot':str(args.snapshot.resolve()) if args.snapshot else None,'reply_server':args.reply_server}
    if send_message(name,message):return 0
    server=MessageServer(name)
    if not server.listening:return 1
    from ui_qt.theme import install_theme
    install_theme(app)
    window=PeakAnalysisWindow(args.session,args.reply_server)
    queued=[]
    def deliver(payload):
        if window.busy:
            queued.append(payload);return
        if payload.get('reply_server'):window.reply_server=payload['reply_server']
        path=Path(payload['snapshot']) if payload.get('snapshot') else None
        if path:
            def loaded(result):
                datasets,_=result
                for d in datasets:d['reply_server']=payload.get('reply_server') or window.reply_server
                # _start owns a worker until finished; queue the merge for its completion.
                window.pending.extend(datasets)
            window._start('Loading data snapshot...',lambda cancel,progress:load_workspace(path,fresh_snapshot=True),loaded)
        window.showNormal();window.raise_();window.activateWindow()
    server.received.connect(deliver)
    timer=QTimer();timer.setInterval(100)
    def drain():
        if queued and not window.busy:deliver(queued.pop(0))
    timer.timeout.connect(drain);timer.start()
    window.show()
    def start():
        if args.session.exists():
            def restored(result):
                datasets,active=result
                # Preserve saved order but select the previously active dataset last.
                window.pending.extend(sorted(datasets,key=lambda d:d['key']==active))
            window._start('Restoring workspace...',lambda cancel,progress:load_workspace(args.session),restored)
        if args.snapshot:queued.append(message)
    QTimer.singleShot(0,start)
    result=app.exec();timer.stop();server.server.close()
    if window.restart_requested:
        from ui_qt.peak_analysis_bridge import launch_arguments
        executable,arguments,cwd=launch_arguments(session_path=window.session_path,reply_server=window.reply_server)
        launched,_=QProcess.startDetached(executable,arguments,cwd)
        if not launched:return 1
    return result


if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
