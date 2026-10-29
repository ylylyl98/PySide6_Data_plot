pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import "NumericFormat.js" as NumericFormat

Item {
    id: editor
    required property var descriptor
    required property var bridge
    required property var colors
    readonly property real numericValue: Number(descriptor.value === undefined ? 0 : descriptor.value)
    readonly property real step: descriptor.step === undefined ? 1 : descriptor.step
    readonly property real lower: descriptor.min === undefined ? -Number.MAX_VALUE : descriptor.min
    readonly property real upper: descriptor.max === undefined ? Number.MAX_VALUE : descriptor.max
    readonly property real minimumReadableWidth: Math.max(92, Math.ceil(metrics.averageCharacterWidth * 9 + 32))
    property bool invalid: false
    property bool dirty: false
    property bool committing: false
    implicitWidth: 130
    implicitHeight: Math.max(30, metrics.height + 10)
    enabled: descriptor.enabled !== false

    FontMetrics { id: metrics; font: input.font }

    function idleText(value) {
        return NumericFormat.display(value, descriptor)
    }
    function refreshText() {
        if (!dirty && !committing)
            input.text = input.activeFocus ? NumericFormat.exact(numericValue) : idleText(numericValue)
    }
    function resetDraft() { dirty = false; invalid = false; refreshText() }
    function parsedDraft() {
        let number = NumericFormat.parse(input.text)
        if (number === undefined || number < lower || number > upper
            || (NumericFormat.policy(descriptor).integer && number !== Math.trunc(number))) return undefined
        return number
    }
    function commit() {
        if (!dirty || descriptor.readOnly) return
        let number = parsedDraft()
        if (number === undefined) { invalid = true; return }
        invalid = false
        dirty = false
        // Idle formatting never changes the finite number that was typed.
        if (number !== numericValue) {
            committing = true
            bridge.setParameter(descriptor.key, number)
            committing = false
        }
        if (!input.activeFocus) refreshText()
    }
    function nudge(direction) {
        if (descriptor.readOnly) return
        let draft = dirty ? parsedDraft() : numericValue
        if (draft === undefined) { invalid = true; return }
        let number = Math.max(lower, Math.min(upper, draft + direction * step))
        if (!isFinite(number) || (NumericFormat.policy(descriptor).integer && number !== Math.trunc(number))) {
            invalid = true
            return
        }
        dirty = false
        invalid = false
        committing = true
        if (number !== numericValue) bridge.setParameter(descriptor.key, number)
        committing = false
        // Do not parse the shortened presentation back into the stored number.
        input.text = idleText(number)
    }
    onNumericValueChanged: refreshText()
    onDescriptorChanged: refreshText()
    Component.onCompleted: refreshText()

    Rectangle {
        anchors.fill: parent
        radius: 4
        color: editor.colors.control
        border.width: input.activeFocus ? 2 : 1
        border.color: editor.invalid ? (editor.colors.error || editor.colors.accent)
                      : input.activeFocus ? editor.colors.focus : editor.colors.border
    }
    Basic.TextField {
        id: input
        objectName: "app_parameter_" + editor.descriptor.key
        anchors.left: parent.left
        anchors.right: steps.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        leftPadding: 7
        rightPadding: 3
        topPadding: 3
        bottomPadding: 3
        selectByMouse: true
        readOnly: editor.descriptor.readOnly === true
        color: enabled ? editor.colors.text : editor.colors.disabled
        selectedTextColor: editor.colors.selectedText
        selectionColor: editor.colors.accent
        inputMethodHints: Qt.ImhFormattedNumbersOnly
        Accessible.name: editor.descriptor.label || editor.descriptor.key
        Accessible.description: editor.invalid ? "Enter a finite number in the allowed range."
                                              : editor.descriptor.description || ""
        background: null
        onEditingFinished: editor.commit()
        onTextEdited: { editor.dirty = true; editor.invalid = false }
        onActiveFocusChanged: editor.refreshText()
        Keys.onUpPressed: editor.nudge(1)
        Keys.onDownPressed: editor.nudge(-1)
        Keys.onEscapePressed: editor.resetDraft()
        Basic.ToolTip.visible: hovered && !activeFocus
        Basic.ToolTip.text: editor.invalid ? "Enter a number between " + editor.lower + " and " + editor.upper
                                         : NumericFormat.exact(editor.numericValue) + (editor.descriptor.description ? "\n" + editor.descriptor.description : "")
    }
    Column {
        id: steps
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        width: 20
        Repeater {
            model: [1, -1]
            delegate: Basic.ToolButton {
                id: stepButton
                required property int modelData
                objectName: "app_parameter_" + (modelData > 0 ? "increase_" : "decrease_") + editor.descriptor.key
                width: steps.width
                height: editor.height / 2
                padding: 0
                focusPolicy: Qt.NoFocus
                enabled: editor.descriptor.readOnly !== true
                autoRepeat: true
                autoRepeatDelay: 400
                autoRepeatInterval: 75
                Accessible.name: (modelData > 0 ? "Increase " : "Decrease ") + (editor.descriptor.label || editor.descriptor.key)
                onClicked: editor.nudge(modelData)
                contentItem: Item {
                    Canvas {
                        anchors.centerIn: parent
                        width: 8
                        height: 5
                        property color stroke: stepButton.enabled ? editor.colors.muted : editor.colors.disabled
                        onStrokeChanged: requestPaint()
                        onPaint: {
                            let context = getContext("2d")
                            context.clearRect(0, 0, width, height)
                            context.strokeStyle = stroke
                            context.lineWidth = 1.2
                            context.beginPath()
                            context.moveTo(1, stepButton.modelData > 0 ? 4 : 1)
                            context.lineTo(4, stepButton.modelData > 0 ? 1 : 4)
                            context.lineTo(7, stepButton.modelData > 0 ? 4 : 1)
                            context.stroke()
                        }
                    }
                }
                background: Rectangle {
                    radius: 3
                    color: stepButton.down ? editor.colors.pressed : stepButton.hovered ? editor.colors.hover : "transparent"
                }
            }
        }
    }
}
