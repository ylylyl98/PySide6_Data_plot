pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic

Basic.ComboBox {
    id: control
    required property var bridge
    required property var colors
    required property string fieldKey
    property var descriptor: bridge.snapshot.choices[fieldKey] || ({})
    property string encodedOptions: ""
    property var stableOptions: []

    function syncOptions() {
        let next = descriptor.options || descriptor.items || []
        let encoded = JSON.stringify(next)
        if (encoded !== encodedOptions) {
            encodedOptions = encoded
            stableOptions = next
        }
    }

    function restoreIndex() {
        // C++ keyboard selection can reach disabled delegate rows. Restore
        // the snapshot binding whenever the bridge rejects a choice.
        currentIndex = Qt.binding(function() {
            return control.descriptor.index === undefined ? -1 : control.descriptor.index
        })
    }

    function activateIndex(index) {
        if (index < 0 || index >= stableOptions.length || stableOptions[index].enabled === false) {
            restoreIndex()
            return
        }
        if (bridge.setChoice(fieldKey, index) === false)
            restoreIndex()
    }

    function stepChoice(direction) {
        for (let next = currentIndex + direction; next >= 0 && next < stableOptions.length; next += direction) {
            if (stableOptions[next].enabled !== false) {
                activateIndex(next)
                return
            }
        }
    }

    objectName: "drr_choice_" + fieldKey
    model: stableOptions
    textRole: "text"
    currentIndex: descriptor.index === undefined ? -1 : descriptor.index
    enabled: descriptor.enabled !== false
    visible: descriptor.visible !== false
    implicitWidth: 140
    implicitHeight: 32
    leftPadding: 10
    rightPadding: 27
    font.pointSize: 9
    hoverEnabled: true
    wheelEnabled: false
    focusPolicy: Qt.StrongFocus
    Accessible.name: descriptor.accessibleName || fieldKey
    Accessible.description: descriptor.tooltip || ""
    onDescriptorChanged: syncOptions()
    Component.onCompleted: syncOptions()
    onActivated: function(index) { activateIndex(index) }
    Keys.priority: Keys.BeforeItem
    Keys.onDownPressed: function(event) {
        if (!popup.visible && !(event.modifiers & Qt.AltModifier)) {
            stepChoice(1)
            event.accepted = true
        } else {
            event.accepted = false
        }
    }
    Keys.onUpPressed: function(event) {
        if (!popup.visible && !(event.modifiers & Qt.AltModifier)) {
            stepChoice(-1)
            event.accepted = true
        } else {
            event.accepted = false
        }
    }
    Basic.ToolTip.visible: hovered && (descriptor.tooltip || "").length > 0
    Basic.ToolTip.text: descriptor.tooltip || ""
    Basic.ToolTip.delay: 650
    contentItem: Text {
        text: control.displayText
        font: control.font
        color: control.enabled ? control.colors.text : control.colors.disabled
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        radius: 6
        color: !control.enabled ? control.colors.controlDisabled
                              : control.down ? control.colors.pressed : control.hovered ? control.colors.hover : control.colors.control
        border.width: control.visualFocus ? 2 : 1
        border.color: control.visualFocus ? control.colors.focus
                                       : !control.enabled ? control.colors.borderDisabled
                                       : control.hovered ? control.colors.borderHover : control.colors.border
    }
    indicator: Canvas {
        x: control.width - width - 9
        y: (control.height - height) / 2
        width: 9
        height: 6
        property color ink: control.enabled ? control.colors.muted : control.colors.disabled
        onInkChanged: requestPaint()
        onPaint: {
            let context = getContext("2d")
            context.reset()
            context.strokeStyle = ink
            context.lineWidth = 1.3
            context.lineCap = "round"
            context.beginPath()
            context.moveTo(1, 1)
            context.lineTo(4.5, 4.5)
            context.lineTo(8, 1)
            context.stroke()
        }
    }
    delegate: Basic.ItemDelegate {
        id: optionDelegate
        required property var modelData
        required property int index
        width: control.popup.width - 8
        implicitHeight: Math.max(32, contentItem.implicitHeight + 12)
        text: modelData.text || ""
        enabled: modelData.enabled !== false
        highlighted: control.highlightedIndex === index
        hoverEnabled: true
        font: control.font
        contentItem: Text {
            text: optionDelegate.text
            font: control.font
            color: optionDelegate.enabled ? control.colors.text : control.colors.disabled
            wrapMode: Text.Wrap
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            radius: 4
            color: optionDelegate.down ? control.colors.pressed
                               : optionDelegate.highlighted ? control.colors.selected
                               : optionDelegate.hovered ? control.colors.hover : "transparent"
            border.width: optionDelegate.visualFocus ? 1 : 0
            border.color: control.colors.focus
        }
    }
    popup: Basic.Popup {
        y: control.height + 3
        width: control.width
        padding: 4
        implicitHeight: Math.min(280, choiceList.contentHeight + 8)
        contentItem: ListView {
            id: choiceList
            clip: true
            model: control.popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex
            boundsBehavior: Flickable.StopAtBounds
            Basic.ScrollIndicator.vertical: Basic.ScrollIndicator { }
        }
        background: Rectangle {
            color: control.colors.control
            border.color: control.colors.border
            radius: 6
        }
    }
}
