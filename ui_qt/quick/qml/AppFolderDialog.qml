pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Dialogs

QtObject {
    id: control

    property string title: "Choose a folder"
    property url currentFolder
    property url selectedFolder
    property int options: FolderDialog.DontUseNativeDialog
    property var parentWindow: null
    property var colors: null
    property bool visible: false
    property var _dialog: null

    signal accepted()
    signal rejected()

    // Qt 6.11's public FolderDialog only applies currentFolder on first show.
    // Fresh instances keep each request's directory without private Qt APIs.
    function open() {
        if (_dialog)
            return
        let properties = {
            title: control.title,
            currentFolder: control.currentFolder,
            options: control.options
        }
        if (control.parentWindow)
            properties.parentWindow = control.parentWindow
        _dialog = implementation.createObject(control, properties)
        if (!_dialog) {
            visible = false
            return
        }
        _dialog.open()
        visible = _dialog.visible
    }

    function close() {
        if (_dialog)
            _dialog.close()
    }

    function reject() {
        if (_dialog)
            _dialog.reject()
    }

    function finish(dialog, wasAccepted) {
        if (_dialog !== dialog)
            return
        currentFolder = dialog.currentFolder
        selectedFolder = dialog.selectedFolder
        _dialog = null
        visible = false
        // The implementation is still emitting its result signal here.
        Qt.callLater(function() { dialog.destroy() })
        if (wasAccepted)
            accepted()
        else
            rejected()
    }

    onVisibleChanged: {
        if (visible && !_dialog)
            open()
        else if (!visible && _dialog && _dialog.visible)
            _dialog.close()
    }

    property Component implementation: Component {
        FolderDialog {
            id: folderImplementation
            AppDialogSelectionPalette { colors: control.colors; dialog: folderImplementation }
            onVisibleChanged: if (control._dialog === folderImplementation) control.visible = visible
            onAccepted: control.finish(folderImplementation, true)
            onRejected: control.finish(folderImplementation, false)
        }
    }
}
