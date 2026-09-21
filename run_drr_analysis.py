"""Standalone entry for the DRR multi-dataset analysis workspace."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from PySide6.QtCore import QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication
from ui_qt.drr_analysis_window import DrrAnalysisWindow
from core.drr_workspace_session import session_directory, load_session


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',nargs='?',default=str(Path.cwd()))
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--session',type=Path,default=session_directory()/'workspace.npz')
    args=parser.parse_args(argv)
    app=QApplication([sys.argv[0]])
    app.setApplicationName('DPTK DRR Analysis');app.setOrganizationName('ylylyl98');app.setStyle('Fusion')
    name='DPTK-DRR-'+hashlib.sha256(str(args.session.resolve()).casefold().encode()).hexdigest()[:24]
    message={'snapshot':str(args.snapshot.resolve()) if args.snapshot else None,'folder':str(Path(args.folder).resolve())}
    socket=QLocalSocket();socket.connectToServer(name)
    if socket.waitForConnected(500):
        socket.write((json.dumps(message)+'\n').encode());socket.flush()
        if socket.bytesToWrite():socket.waitForBytesWritten(3000)
        socket.disconnectFromServer()
        return 0
    server=QLocalServer();server.setSocketOptions(QLocalServer.UserAccessOption)
    if not server.listen(name):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(None,'DRR Analysis','Another analysis process is starting. Please try again.')
        return 1
    from ui_qt.theme import install_theme
    install_theme(app)
    window=DrrAnalysisWindow(start_folder=args.folder,session_path=args.session)
    if args.session.exists():
        try:window.restore_session(args.session)
        except ValueError as exc:
            window.session_path=None
            window.notice.setText(f'{exc} Original file preserved; use Save workspace to choose another file.')
    def deliver(payload):
        try:
            folder=payload.get('folder')
            if folder and str(Path(window.start_folder).resolve())!=folder:
                window.start_folder=folder
                window.processed_groups.clear()
                window.refresh_processed_groups()
            if payload.get('snapshot'):
                path=Path(payload['snapshot']);datasets,_,_=load_session(path)
                for dataset in datasets:window.add_dataset(dataset)
                saved=window.save_workspace(quiet=True)
                window.notice.setText('Received current DRR snapshot.')
                if saved and path.parent.resolve()==(session_directory()/'inbox').resolve():path.unlink(missing_ok=True)
        except Exception as exc:window.notice.setText(f'Cannot import snapshot: {exc}')
        window.showNormal();window.raise_();window.activateWindow()
    connections=[]
    def connected():
        while server.hasPendingConnections():
            client=server.nextPendingConnection();connections.append(client)
            buffer=bytearray()
            def receive(client=client,buffer=buffer):
                buffer.extend(bytes(client.readAll()))
                if len(buffer)>65536:client.abort();return
                if b'\n' in buffer:
                    try:deliver(json.loads(bytes(buffer).split(b'\n',1)[0]))
                    except (ValueError,TypeError):pass
                    buffer.clear();client.disconnectFromServer()
            client.readyRead.connect(receive)
            client.disconnected.connect(lambda client=client:(connections.remove(client) if client in connections else None,client.deleteLater()))
            receive()
    server.newConnection.connect(connected)
    window.show()
    if args.snapshot:QTimer.singleShot(0,lambda:deliver(message))
    return app.exec()


if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
