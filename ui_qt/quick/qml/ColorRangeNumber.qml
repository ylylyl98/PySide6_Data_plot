pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: control
    required property var controller
    required property var colors
    required property var descriptor
    required property string endpoint
    required property int contextVersion
    property var preview: ({})
    property bool dirty: false
    property bool invalid: false
    property bool committing: false
    property bool ready: false
    readonly property string groupId: descriptor.id || ""
    readonly property bool minimum: endpoint === "low"
    readonly property string canonicalText: (minimum ? descriptor.lowText : descriptor.highText) || ""
    readonly property string previewText: (minimum ? preview.lowText : preview.highText) || canonicalText
    readonly property string identity: groupId + ":" + endpoint + ":" + contextVersion
    spacing: 3
    objectName: "color_sidebar_number_" + endpoint + "_" + groupId

    function sync() {
        if (ready && !dirty && !committing) input.text = previewText
    }
    function cancel() {
        let clearError = invalid
        dirty = false
        invalid = false
        if (clearError && controller) controller.clearError()
        sync()
    }
    function commit() {
        if (!dirty) return true
        committing = true
        let accepted = controller.commitNumber(groupId, endpoint, input.text)
        committing = false
        invalid = !accepted
        if (accepted) { dirty = false; sync() }
        return accepted
    }
    onPreviewTextChanged: sync()
    onCanonicalTextChanged: sync()
    onIdentityChanged: if (ready && !committing) cancel()
    Component.onCompleted: { ready = true; sync() }

    RowLayout {
        Layout.fillWidth: true
        spacing: 4
        Text {
            Layout.fillWidth: true
            text: control.minimum ? "Min" : "Max"
            color: control.colors.muted
        }
        AppButton {
            objectName: "color_sidebar_fix_" + (control.minimum ? "min_" : "max_") + control.groupId
            colors: control.colors
            compact: true
            text: "Fix"
            checkable: true
            checked: (control.minimum ? control.descriptor.fixedLow : control.descriptor.fixedHigh) === true
            enabled: control.descriptor.enabled !== false
            description: "Fix " + control.descriptor.title + (control.minimum ? " minimum" : " maximum") + " for sliders and Auto; exact edits remain allowed"
            onClicked: control.controller.setFixed(control.groupId, control.endpoint, checked)
        }
    }
    Basic.TextField {
        id: input
        objectName: "color_sidebar_" + (control.minimum ? "min_" : "max_") + control.groupId
        Layout.fillWidth: true
        Layout.minimumWidth: 70
        implicitHeight: 32
        selectByMouse: true
        persistentSelection: false
        font.pointSize: 9
        color: control.enabled ? control.colors.text : control.colors.disabled
        selectionColor: control.colors.accent
        selectedTextColor: control.colors.selectedText
        enabled: control.descriptor.enabled !== false
        inputMethodHints: Qt.ImhFormattedNumbersOnly
        Accessible.name: control.descriptor.title + (control.minimum ? " color minimum" : " color maximum")
        Accessible.description: control.invalid ? "Invalid value; correct it or press Escape to cancel."
                               : control.dirty ? "Pending value. Enter or leave this field to apply. Escape cancels."
                               : "Synchronized with the color slider. Decimal and scientific notation accepted."
        onTextEdited: { control.dirty = true; control.invalid = false }
        onAccepted: control.commit()
        onActiveFocusChanged: {
            if (activeFocus) selectAll()
            else control.commit()
        }
        Keys.onEscapePressed: { control.cancel(); selectAll() }
        background: Rectangle {
            color: control.colors.control
            radius: 4
            border.width: input.activeFocus ? 2 : 1
            border.color: control.invalid ? control.colors.error : input.activeFocus ? control.colors.focus : control.colors.border
        }
    }
}
