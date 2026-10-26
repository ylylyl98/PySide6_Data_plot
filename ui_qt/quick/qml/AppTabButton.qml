pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic

Basic.TabButton {
    id: control
    required property var colors
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus
    contentItem: Text {
        text: control.text
        font: control.font
        color: !control.enabled ? control.colors.disabled
             : control.checked ? control.colors.accentText : control.colors.text
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        implicitHeight: 40
        color: !control.enabled ? control.colors.controlDisabled
             : control.down ? control.colors.pressed
             : control.checked ? control.colors.selected
             : control.hovered ? control.colors.hover : control.colors.control
        border.width: control.visualFocus ? 2 : control.checked ? 1 : 0
        border.color: control.visualFocus ? control.colors.focus : control.colors.selectedBorder
    }
}
