pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic

Basic.ComboBox {
    id: control
    required property var colors
    hoverEnabled: true

    palette.window: colors.control
    palette.base: colors.control
    palette.text: enabled ? colors.text : colors.disabled
    palette.buttonText: enabled ? colors.text : colors.disabled
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.mid: colors.border
    palette.dark: enabled ? colors.muted : colors.disabled

    // Keep Qt's popup, model, editable text, scrolling and keyboard handling.
    // Only replace the mismatched Basic selection colors.
    delegate: AppItemDelegate {
        required property var model
        required property int index
        colors: control.colors
        width: ListView.view.width
        text: model[control.textRole]
        font.weight: control.currentIndex === index ? Font.DemiBold : Font.Normal
        highlighted: control.highlightedIndex === index
        hoverEnabled: control.hoverEnabled
    }

    background: Rectangle {
        implicitWidth: 140
        implicitHeight: 40
        color: !control.enabled ? control.colors.controlDisabled
             : control.down ? control.colors.pressed
             : control.hovered ? control.colors.hover : control.colors.control
        border.width: control.visualFocus ? 2 : 1
        border.color: control.visualFocus ? control.colors.focus
                     : control.enabled ? control.colors.border : control.colors.borderDisabled
        visible: !control.flat || control.down
    }
}
