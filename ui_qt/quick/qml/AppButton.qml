pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic

Basic.ToolButton {
    id: control
    required property var colors
    property string description: text
    property bool primary: false
    property bool compact: false
    implicitHeight: 30
    implicitWidth: Math.max(compact ? 28 : 54, label.implicitWidth + (compact ? 14 : 20))
    padding: compact ? 5 : 8
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus
    Accessible.name: description
    Accessible.description: description
    Basic.ToolTip.visible: hovered && description.length > 0
    Basic.ToolTip.text: description
    Basic.ToolTip.delay: 650
    contentItem: Text {
        id: label
        text: control.text
        font: control.font
        color: !control.enabled ? control.colors.disabled
             : control.primary ? control.colors.selectedText
             : control.checked ? control.colors.accentText : control.colors.text
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        radius: 4
        color: control.primary ? control.colors.accent
             : control.down ? control.colors.pressed
             : control.checked ? control.colors.selected
             : control.hovered ? control.colors.hover : "transparent"
        border.width: control.visualFocus ? 2 : control.checked || control.primary ? 1 : 0
        border.color: control.visualFocus ? control.colors.focus
                    : control.checked ? control.colors.selectedBorder : control.colors.accent
    }
}
