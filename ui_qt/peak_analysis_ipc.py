"""Small, current-user local-socket messages; numeric data stays in NPZ files."""
import json
from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


def send_message(name, message, timeout=500):
    if not name:
        return False
    socket = QLocalSocket(); socket.connectToServer(name)
    if not socket.waitForConnected(timeout):
        return False
    socket.write((json.dumps(message)+'\n').encode('utf-8')); socket.flush()
    sent = not socket.bytesToWrite() or socket.waitForBytesWritten(2000)
    socket.disconnectFromServer()
    return sent


class MessageServer(QObject):
    received = Signal(dict)

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.server = QLocalServer(self); self.server.setSocketOptions(QLocalServer.UserAccessOption)
        self.listening = self.server.listen(name); self.clients = []
        self.server.newConnection.connect(self.accept)

    def accept(self):
        while self.server.hasPendingConnections():
            client=self.server.nextPendingConnection();self.clients.append(client);buffer=bytearray()
            def read(client=client, buffer=buffer):
                buffer.extend(bytes(client.readAll()))
                if len(buffer)>65536:
                    client.abort();return
                if b'\n' in buffer:
                    try:
                        value=json.loads(bytes(buffer).split(b'\n',1)[0])
                        if isinstance(value,dict):self.received.emit(value)
                    except (ValueError,UnicodeError):
                        pass
                    client.disconnectFromServer()
            client.readyRead.connect(read)
            client.disconnected.connect(lambda c=client:(self.clients.remove(c) if c in self.clients else None,c.deleteLater()))
            read()
