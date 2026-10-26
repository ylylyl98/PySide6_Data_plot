pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic

Basic.ItemDelegate {
    id: control
    required property var colors
    hoverEnabled: true

    // Basic uses highlightedText over palette.light. Keep the foreground
    // paired with our subtle selection surface, including high contrast.
    palette.text: enabled ? colors.text : colors.disabled
    palette.highlightedText: enabled ? colors.text : colors.disabled

    background: Rectangle {
        implicitWidth: 100
        implicitHeight: 40
        color: !control.enabled ? control.colors.controlDisabled
             : control.down ? control.colors.pressed
             : control.highlighted ? control.colors.selected
             : control.hovered ? control.colors.hover : control.colors.control
        border.width: control.visualFocus ? 2 : control.highlighted ? 1 : 0
        border.color: control.visualFocus ? control.colors.focus : control.colors.selectedBorder
    }
}
