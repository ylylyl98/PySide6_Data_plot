pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

AppItemDelegate {
    id: control
    required property var entry
    checked: entry.selected === true
    checkable: false
    focusPolicy: Qt.StrongFocus
    default property alias details: labels.data
    signal selectionRequested(bool selected)
    padding: 8
    text: entry.name
    highlighted: checked
    Accessible.name: "Add " + entry.source + " to selection"
    // Filename gestures always add. Only the checkbox or Space toggles, so
    // repeated clicks cannot remove and invalidate an existing reference.
    onClicked: selectionRequested(true)
    onDoubleClicked: selectionRequested(true)
    Keys.onSpacePressed: event => { selectionRequested(!checked); event.accepted = true }
    Keys.onReturnPressed: event => { selectionRequested(true); event.accepted = true }
    Keys.onEnterPressed: event => { selectionRequested(true); event.accepted = true }
    Basic.ToolTip.visible: hovered
    Basic.ToolTip.text: entry.source + "\n" + (entry.detail || "")
        + "\nClick or double-click the filename to add. Uncheck to remove."
    Basic.ToolTip.delay: 650
    background: Rectangle {
        radius: 5
        color: control.checked ? control.colors.selected : control.hovered ? control.colors.hover : control.colors.control
        border.width: control.visualFocus ? 2 : 1
        border.color: control.visualFocus ? control.colors.focus : control.checked ? control.colors.accent : control.colors.divider
    }
    contentItem: RowLayout {
        spacing: 6
        Basic.CheckBox {
            objectName: control.objectName + "_check"
            checked: control.checked
            padding: 0
            Accessible.name: "Select " + control.entry.source
            onToggled: control.selectionRequested(checked)
        }
        ColumnLayout {
            id: labels
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            spacing: 5
        }
    }
}
