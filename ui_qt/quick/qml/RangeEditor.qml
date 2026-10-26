import QtQuick
import QtQuick.Layouts

ColumnLayout {
    id: control
    required property var bridge
    required property var colors
    required property string title
    required property string minimumKey
    required property string maximumKey
    required property string autoAction
    property bool showFix: true
    spacing: 5

    objectName: "drr_range_" + minimumKey

    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        Text {
            Layout.fillWidth: true
            text: control.title === "X" || control.title === "Y" ? control.title + " axis" : control.title + " limits"
            color: control.colors.text
            font.pointSize: 9
            font.weight: Font.DemiBold
        }
        DisplayButton {
            objectName: "drr_action_" + control.autoAction
            colors: control.colors
            text: "Auto"
            Layout.minimumWidth: 56
            implicitHeight: 26
            subtle: true
            enabled: (control.bridge.snapshot.actions[control.autoAction] || {}).enabled !== false
            tooltip: (control.bridge.snapshot.actions[control.autoAction] || {}).tooltip || "Set limits automatically, retaining fixed values."
            Accessible.name: "Automatic " + control.title + " limits"
            onClicked: control.bridge.invoke(control.autoAction)
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: 10
        ColumnLayout {
            Layout.fillWidth: true
            Layout.preferredWidth: 1
            Layout.alignment: Qt.AlignTop
            spacing: 3
            RowLayout {
                Layout.fillWidth: true
                Layout.minimumHeight: 22
                spacing: 4
                Text {
                    Layout.fillWidth: true
                    text: "Min"
                    color: control.colors.muted
                    font.pointSize: 8
                }
                FlagEditor {
                    bridge: control.bridge
                    colors: control.colors
                    fieldKey: "fix." + control.minimumKey
                    compact: true
                    visible: control.showFix && descriptor.visible !== false
                    Accessible.name: "Fix " + control.title + " minimum"
                }
            }
            NumberEditor {
                Layout.fillWidth: true
                bridge: control.bridge
                colors: control.colors
                fieldKey: control.minimumKey
                accessibleLabel: control.title + " minimum"
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.preferredWidth: 1
            Layout.alignment: Qt.AlignTop
            spacing: 3
            RowLayout {
                Layout.fillWidth: true
                Layout.minimumHeight: 22
                spacing: 4
                Text {
                    Layout.fillWidth: true
                    text: "Max"
                    color: control.colors.muted
                    font.pointSize: 8
                }
                FlagEditor {
                    bridge: control.bridge
                    colors: control.colors
                    fieldKey: "fix." + control.maximumKey
                    compact: true
                    visible: control.showFix && descriptor.visible !== false
                    Accessible.name: "Fix " + control.title + " maximum"
                }
            }
            NumberEditor {
                Layout.fillWidth: true
                bridge: control.bridge
                colors: control.colors
                fieldKey: control.maximumKey
                accessibleLabel: control.title + " maximum"
            }
        }
    }
}
