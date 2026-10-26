import QtQuick
import QtQuick.Controls.Basic as Basic

Basic.Button {
    id: control
    required property var colors
    property string tooltip: ""
    property bool subtle: false

    implicitHeight: 32
    implicitWidth: Math.max(56, contentItem.implicitWidth + 24)
    padding: 10
    topPadding: 4
    bottomPadding: 4
    font.pointSize: 9
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus
    Accessible.name: text
    Accessible.description: tooltip
    Basic.ToolTip.visible: hovered && tooltip.length > 0
    Basic.ToolTip.text: tooltip
    Basic.ToolTip.delay: 650
    contentItem: Text {
        text: control.text
        font: control.font
        color: !control.enabled ? control.colors.disabled
                                : control.subtle ? control.colors.accentText : control.colors.text
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        radius: 6
        color: !control.enabled ? control.subtle ? "transparent" : control.colors.controlDisabled
                            : control.down ? control.colors.pressed
                            : control.hovered ? control.colors.hover
                                              : control.subtle ? "transparent" : control.colors.control
        border.width: control.visualFocus ? 2 : control.subtle ? 0 : 1
        border.color: control.visualFocus ? control.colors.focus
                                         : !control.enabled ? control.colors.borderDisabled
                                         : control.hovered ? control.colors.borderHover : control.colors.border
    }
}
