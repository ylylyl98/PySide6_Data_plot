pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Dialogs

FileDialog {
    id: control
    required property var colors

    AppDialogSelectionPalette {
        colors: control.colors
        dialog: control
    }
}
