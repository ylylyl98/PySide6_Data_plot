import QtQuick
import QtQuick.Controls.Basic as Basic

Basic.CheckBox {
    id: control
    required property var bridge
    required property var colors
    required property string fieldKey
    property string label: ""
    property bool compact: false
    property var descriptor: bridge.snapshot.flags[fieldKey] || ({})

    objectName: "drr_flag_" + fieldKey
    text: label || descriptor.text || "Fix"
    checked: descriptor.checked === true
    enabled: descriptor.enabled !== false
    visible: descriptor.visible !== false
    implicitWidth: contentItem.implicitWidth + leftPadding + rightPadding
    implicitHeight: compact ? 22 : Math.max(28, contentItem.implicitHeight + 6)
    spacing: compact ? 5 : 8
    leftPadding: compact ? 6 : 0
    rightPadding: compact ? 7 : 0
    topPadding: 2
    bottomPadding: 2
    font.pointSize: compact ? 8 : 9
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus
    Accessible.name: descriptor.accessibleName || text
    Accessible.description: descriptor.tooltip || ""
    onToggled: bridge.setFlag(fieldKey, checked)
    Basic.ToolTip.visible: hovered && (descriptor.tooltip || "").length > 0
    Basic.ToolTip.text: descriptor.tooltip || ""
    Basic.ToolTip.delay: 650
    background: Rectangle {
        radius: 5
        color: !control.compact || !control.enabled ? "transparent"
                             : control.down ? control.colors.pressed
                             : control.checked ? control.colors.selected
                             : control.hovered ? control.colors.hover : "transparent"
        border.width: control.visualFocus ? 2 : control.compact && control.checked ? 1 : 0
        border.color: control.visualFocus ? control.colors.focus : control.colors.selectedBorder
    }
    indicator: Rectangle {
        x: control.compact ? 6 : 0
        y: Math.round((control.height - height) / 2)
        width: control.compact ? 12 : 16
        height: width
        radius: 3
        color: control.checked ? control.colors.accent
                               : control.down ? control.colors.pressed
                                              : control.hovered ? control.colors.hover : control.colors.control
        border.width: control.visualFocus && !control.compact ? 2 : 1
        border.color: control.visualFocus && !control.compact ? control.colors.focus
                                                          : control.checked ? control.colors.accent
                                                          : control.hovered ? control.colors.borderHover : control.colors.border
        opacity: control.enabled ? 1 : 0.5
        Canvas {
            anchors.fill: parent
            visible: control.checked
            property color ink: control.colors.selectedText
            onInkChanged: requestPaint()
            onVisibleChanged: if (visible) requestPaint()
            onPaint: {
                let context = getContext("2d")
                context.reset()
                context.strokeStyle = ink
                context.lineWidth = control.compact ? 1.4 : 1.8
                context.lineCap = "round"
                context.lineJoin = "round"
                context.beginPath()
                context.moveTo(width * 0.25, height * 0.5)
                context.lineTo(width * 0.44, height * 0.69)
                context.lineTo(width * 0.75, height * 0.31)
                context.stroke()
            }
        }
        Rectangle {
            anchors.fill: parent
            anchors.margins: -3
            radius: 5
            color: "transparent"
            border.width: control.visualFocus && !control.compact ? 1 : 0
            border.color: control.colors.focus
        }
    }
    contentItem: Text {
        leftPadding: control.indicator.width + control.spacing
        text: control.text
        font: control.font
        color: !control.enabled ? control.colors.disabled
                                : control.compact && control.checked ? control.colors.accentText : control.colors.text
        wrapMode: Text.Wrap
        verticalAlignment: Text.AlignVCenter
    }
}
