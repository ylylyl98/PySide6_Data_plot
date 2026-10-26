pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: rail
    required property var colors
    required property var workflows
    required property string currentWorkflow
    signal selected(string key)
    implicitWidth: 104
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "Workflow navigation"

    function moveFocus(index, direction) {
        let destination = direction === 0 ? 0
                        : direction === 2 ? workflows.length - 1
                        : Math.max(0, Math.min(workflows.length - 1, index + direction))
        let item = destinations.itemAt(destination)
        if (item) item.focusDestination(Qt.ShortcutFocusReason)
    }
    function ensureFocusVisible(item) {
        let flick = navigationScroll.contentItem as Flickable
        let position = item.mapToItem(railList, 0, 0)
        if (position.y < flick.contentY + 4) flick.contentY = Math.max(0, position.y - 4)
        else if (position.y + item.height > flick.contentY + flick.height - 4)
            flick.contentY = Math.min(Math.max(0, flick.contentHeight - flick.height), position.y + item.height - flick.height + 4)
    }

    Rectangle {
        anchors.right: parent.right
        width: 1
        height: parent.height
        color: rail.colors.divider
    }
    Basic.ScrollView {
        id: navigationScroll
        anchors.fill: parent
        anchors.margins: 8
        contentWidth: availableWidth
        clip: true
        Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
        ColumnLayout {
            id: railList
            width: parent.width
            height: Math.max(implicitHeight, navigationScroll.availableHeight)
            spacing: 5
            Text {
                Layout.fillWidth: true
                Layout.topMargin: 8
                Layout.bottomMargin: 9
                text: "ANALYZE"
                font.pixelSize: 10
                font.weight: Font.DemiBold
                color: rail.colors.muted
                horizontalAlignment: Text.AlignHCenter
            }
            Repeater {
                id: destinations
                model: rail.workflows
                delegate: ColumnLayout {
                    id: destination
                    required property int index
                    required property var modelData
                    Layout.fillWidth: true
                    Layout.fillHeight: modelData.key === "slides"
                    spacing: 5
                    function focusDestination(reason) {
                        button.forceActiveFocus(reason)
                        rail.ensureFocusVisible(button)
                    }
                    Item {
                        Layout.fillHeight: true
                        visible: destination.modelData.key === "slides"
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.topMargin: 10
                        Layout.bottomMargin: 5
                        implicitHeight: 1
                        color: rail.colors.divider
                        visible: destination.modelData.key === "slides"
                    }
                    AppButton {
                        id: button
                        objectName: "app_workflow_" + destination.modelData.key
                        Layout.fillWidth: true
                        implicitHeight: destination.modelData.key === "mcd_peak_shift" ? 52 : 42
                        colors: rail.colors
                        text: destination.modelData.label
                        description: destination.modelData.description
                        checkable: true
                        checked: rail.currentWorkflow === destination.modelData.key
                        focusPolicy: checked ? Qt.StrongFocus : Qt.ClickFocus
                        onClicked: rail.selected(destination.modelData.key)
                        onActiveFocusChanged: if (activeFocus) rail.ensureFocusVisible(button)
                        contentItem: Text {
                            text: button.text
                            font.pointSize: button.font.pointSize
                            font.weight: button.checked ? Font.DemiBold : Font.Normal
                            color: button.checked ? rail.colors.accentText : rail.colors.text
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            wrapMode: Text.WordWrap
                        }
                        Keys.onPressed: function(event) {
                            if (event.key === Qt.Key_Down || event.key === Qt.Key_Right) {
                                rail.moveFocus(destination.index, 1); event.accepted = true
                            } else if (event.key === Qt.Key_Up || event.key === Qt.Key_Left) {
                                rail.moveFocus(destination.index, -1); event.accepted = true
                            } else if (event.key === Qt.Key_Home) {
                                rail.moveFocus(destination.index, 0); event.accepted = true
                            } else if (event.key === Qt.Key_End) {
                                rail.moveFocus(destination.index, 2); event.accepted = true
                            } else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) {
                                rail.selected(destination.modelData.key); event.accepted = true
                            }
                        }
                    }
                }
            }
        }
    }
}
