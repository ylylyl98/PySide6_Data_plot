pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic

Item {
    id: overlay
    required property var bridge
    required property var colors
    required property var state
    property bool dragging: false
    readonly property real footerHeight: spectrumToggle.implicitHeight + 16
    readonly property real plotHeight: Math.max(0, height - footerHeight)
    readonly property bool spectrumVisible: state.spectrumVisible !== false
    readonly property real dividerY: state.dividerY === undefined ? 0.7 : state.dividerY
    readonly property real leftFraction: state.dividerLeft === undefined ? 0.12 : state.dividerLeft
    readonly property real rightFraction: state.dividerRight === undefined ? 0.94 : state.dividerRight

    function adjust(direction) {
        let fraction = state.heatmapFraction === undefined ? 0.7 : state.heatmapFraction
        bridge.setParameter("heatmapFraction", Math.max(0.15, Math.min(0.9, fraction + direction * 0.02)))
    }

    Basic.Control {
        id: handle
        objectName: "app_inspection_divider"
        x: overlay.width * overlay.leftFraction
        y: overlay.plotHeight * overlay.dividerY - height / 2
        width: Math.max(0, overlay.width * (overlay.rightFraction - overlay.leftFraction))
        height: 10
        visible: overlay.spectrumVisible
        focusPolicy: Qt.StrongFocus
        Accessible.role: Accessible.Separator
        Accessible.name: "Resize heatmap and spectrum"
        Accessible.description: "Drag or use arrow keys to resize the heatmap and spectrum. Home restores the 70 percent heatmap layout."
        background: Rectangle {
            anchors.verticalCenter: parent.verticalCenter
            height: handle.activeFocus || dragArea.containsMouse || overlay.dragging ? 3 : 1
            color: handle.activeFocus || dragArea.containsMouse || overlay.dragging ? overlay.colors.accent : overlay.colors.divider
        }
        Keys.onUpPressed: overlay.adjust(1)
        Keys.onDownPressed: overlay.adjust(-1)
        Keys.onLeftPressed: overlay.adjust(-1)
        Keys.onRightPressed: overlay.adjust(1)
        Keys.onPressed: function(event) {
            if (event.key === Qt.Key_Home) {
                overlay.bridge.setParameter("heatmapFraction", 0.7)
                event.accepted = true
            }
        }
        MouseArea {
            id: dragArea
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.SplitVCursor
            preventStealing: true
            onPressed: {
                overlay.dragging = true
                handle.forceActiveFocus(Qt.MouseFocusReason)
            }
            onPositionChanged: function(mouse) {
                if (!pressed) return
                let position = mapToItem(overlay, mouse.x, mouse.y)
                overlay.bridge.setParameter("dividerY", Math.max(0.05, Math.min(0.95, position.y / Math.max(1, overlay.plotHeight))))
            }
            onReleased: overlay.dragging = false
            onCanceled: overlay.dragging = false
        }
        Basic.ToolTip.visible: dragArea.containsMouse && !overlay.dragging
        Basic.ToolTip.text: "Drag to resize heatmap / spectrum · Home restores 70 / 30"
        Basic.ToolTip.delay: 700
    }
    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: overlay.footerHeight
        color: overlay.colors.surface
        Rectangle { width: parent.width; height: 1; color: overlay.colors.divider }
    }
    AppButton {
        id: spectrumToggle
        objectName: "app_toggle_spectrum"
        colors: overlay.colors
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        width: Math.min(implicitWidth, Math.max(0, overlay.width - 16))
        primary: !overlay.spectrumVisible
        text: overlay.spectrumVisible ? "Hide spectrum" : "Show spectrum"
        description: overlay.spectrumVisible ? "Hide the spectrum and expand the heatmap" : "Show the spectrum below the heatmap"
        onClicked: overlay.bridge.invoke("toggle-spectrum")
        background: Rectangle {
            radius: 4
            color: spectrumToggle.primary ? overlay.colors.accent
                 : spectrumToggle.down ? overlay.colors.pressed
                 : spectrumToggle.hovered ? overlay.colors.hover : overlay.colors.control
            border.width: spectrumToggle.visualFocus ? 2 : 1
            border.color: spectrumToggle.visualFocus ? overlay.colors.focus
                        : spectrumToggle.primary ? overlay.colors.accent : overlay.colors.border
        }
    }
}
