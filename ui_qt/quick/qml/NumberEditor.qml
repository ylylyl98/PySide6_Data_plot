import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: control
    required property var bridge
    required property var colors
    required property string fieldKey
    property string accessibleLabel: ""
    property var descriptor: bridge.snapshot.numbers[fieldKey] || ({})
    property bool dirty: false
    property string localError: ""
    property bool committing: false
    property bool componentReady: false
    readonly property string errorText: localError || (dirty ? "" : descriptor.error || "")
    readonly property alias textField: input

    objectName: "drr_number_" + fieldKey
    implicitWidth: 130
    spacing: 3
    enabled: descriptor.enabled !== false
    visible: descriptor.visible !== false

    function refreshText() {
        if (!dirty && !localError && !committing)
            input.text = input.activeFocus ? String(descriptor.editText || descriptor.text || "")
                                          : String(descriptor.text || "")
    }

    function commit() {
        if (!dirty)
            return !localError
        committing = true
        let accepted = bridge.setNumber(fieldKey, input.text)
        committing = false
        if (accepted) {
            dirty = false
            localError = ""
            refreshText()
        } else {
            localError = descriptor.error || "Enter a finite value within the allowed range."
        }
        return accepted
    }

    function cancelDraft() {
        dirty = false
        localError = ""
        bridge.setEditing(fieldKey, false)
        // The unchanged canonical value clears backend validation feedback
        // without emitting a numeric or finish operation.
        bridge.setNumber(fieldKey, String(descriptor.editText || descriptor.text || ""))
        refreshText()
    }

    function settleDraft() {
        if (dirty) {
            if (!commit())
                cancelDraft()
        } else if (localError || descriptor.error) {
            cancelDraft()
        }
        bridge.setEditing(fieldKey, false)
    }

    function step(direction) {
        let accepted = dirty ? bridge.stepNumber(fieldKey, direction, input.text)
                             : bridge.stepNumber(fieldKey, direction)
        if (accepted) {
            dirty = false
            localError = ""
            refreshText()
        } else {
            localError = descriptor.error || "Enter a valid value before stepping."
        }
    }

    onDescriptorChanged: refreshText()
    onVisibleChanged: {
        if (componentReady && !visible)
            settleDraft()
    }
    Component.onCompleted: {
        componentReady = true
        refreshText()
    }

    Connections {
        target: control.bridge
        ignoreUnknownSignals: true
        function onDeactivating() { control.settleDraft() }
    }

    Rectangle {
        id: shell
        Layout.fillWidth: true
        implicitHeight: 34
        radius: 6
        color: control.enabled ? control.colors.control : control.colors.controlDisabled
        border.width: input.activeFocus || increase.visualFocus || decrease.visualFocus ? 2 : 1
        border.color: control.errorText ? control.colors.error
                                       : input.activeFocus || increase.visualFocus || decrease.visualFocus
                                         ? control.colors.focus
                                         : !control.enabled ? control.colors.borderDisabled
                                         : fieldHover.hovered ? control.colors.borderHover : control.colors.border

        HoverHandler { id: fieldHover }

        Basic.TextField {
            id: input
            objectName: "drr_input_" + control.fieldKey
            anchors.left: parent.left
            anchors.right: stepButtons.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            leftPadding: 10
            rightPadding: 4
            topPadding: 3
            bottomPadding: 3
            selectByMouse: true
            persistentSelection: false
            font.pointSize: 9
            color: control.enabled ? control.colors.text : control.colors.disabled
            selectionColor: control.colors.accent
            selectedTextColor: control.colors.selectedText
            background: null
            inputMethodHints: Qt.ImhFormattedNumbersOnly
            Accessible.name: control.accessibleLabel || control.descriptor.accessibleName || control.fieldKey
            Accessible.description: control.errorText || (control.dirty ? "Pending value. Press Enter or leave this field to apply; Escape cancels." : control.descriptor.tooltip || "")
            onTextEdited: {
                control.dirty = true
                control.localError = ""
                control.bridge.setEditing(control.fieldKey, true)
            }
            onActiveFocusChanged: {
                if (activeFocus) {
                    control.bridge.setEditing(control.fieldKey, control.dirty)
                    control.refreshText()
                    selectAll()
                } else {
                    control.commit()
                    control.bridge.setEditing(control.fieldKey, false)
                    control.refreshText()
                }
            }
            onAccepted: control.commit()
            Keys.onUpPressed: function(event) { control.step(1); event.accepted = true }
            Keys.onDownPressed: function(event) { control.step(-1); event.accepted = true }
            Keys.onEscapePressed: function(event) {
                control.cancelDraft()
                selectAll()
                event.accepted = true
            }
            Basic.ToolTip.visible: hovered && (control.descriptor.tooltip || "").length > 0
            Basic.ToolTip.text: control.descriptor.tooltip || ""
            Basic.ToolTip.delay: 650
        }

        Column {
            id: stepButtons
            anchors.right: parent.right
            anchors.rightMargin: 1
            anchors.top: parent.top
            anchors.topMargin: 1
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 1
            width: 24

            Basic.Button {
                id: increase
                objectName: "drr_increase_" + control.fieldKey
                width: parent.width
                height: parent.height / 2
                focusPolicy: Qt.TabFocus
                hoverEnabled: true
                autoRepeat: true
                autoRepeatDelay: 350
                autoRepeatInterval: 75
                Accessible.name: "Increase " + (control.accessibleLabel || control.descriptor.accessibleName || control.fieldKey)
                onClicked: control.step(1)
                background: Rectangle {
                    radius: 4
                    color: increase.down ? control.colors.pressed : increase.hovered ? control.colors.hover : "transparent"
                    border.width: increase.visualFocus ? 1 : 0
                    border.color: control.colors.focus
                }
                contentItem: Canvas {
                    property color ink: control.enabled ? control.colors.muted : control.colors.disabled
                    onInkChanged: requestPaint()
                    onPaint: {
                        let context = getContext("2d")
                        context.reset()
                        context.strokeStyle = ink
                        context.lineWidth = 1.2
                        context.lineCap = "round"
                        context.beginPath()
                        context.moveTo(width / 2 - 3, height / 2 + 1.5)
                        context.lineTo(width / 2, height / 2 - 1.5)
                        context.lineTo(width / 2 + 3, height / 2 + 1.5)
                        context.stroke()
                    }
                }
            }
            Basic.Button {
                id: decrease
                objectName: "drr_decrease_" + control.fieldKey
                width: parent.width
                height: parent.height / 2
                focusPolicy: Qt.TabFocus
                hoverEnabled: true
                autoRepeat: true
                autoRepeatDelay: 350
                autoRepeatInterval: 75
                Accessible.name: "Decrease " + (control.accessibleLabel || control.descriptor.accessibleName || control.fieldKey)
                onClicked: control.step(-1)
                background: Rectangle {
                    radius: 4
                    color: decrease.down ? control.colors.pressed : decrease.hovered ? control.colors.hover : "transparent"
                    border.width: decrease.visualFocus ? 1 : 0
                    border.color: control.colors.focus
                }
                contentItem: Canvas {
                    property color ink: control.enabled ? control.colors.muted : control.colors.disabled
                    onInkChanged: requestPaint()
                    onPaint: {
                        let context = getContext("2d")
                        context.reset()
                        context.strokeStyle = ink
                        context.lineWidth = 1.2
                        context.lineCap = "round"
                        context.beginPath()
                        context.moveTo(width / 2 - 3, height / 2 - 1.5)
                        context.lineTo(width / 2, height / 2 + 1.5)
                        context.lineTo(width / 2 + 3, height / 2 - 1.5)
                        context.stroke()
                    }
                }
            }
        }
    }
    Text {
        Layout.fillWidth: true
        text: control.errorText
        visible: text.length > 0
        color: control.colors.error
        font.pointSize: 8
        wrapMode: Text.Wrap
        Accessible.role: Accessible.StaticText
        Accessible.name: text
    }
}
