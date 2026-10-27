pragma ComponentBehavior: Bound

import QtQuick

QtObject {
    id: helper
    required property var colors
    required property var dialog
    readonly property var targetWindow: dialog ? dialog.parentWindow : null
    readonly property bool active: !!colors && !!dialog && dialog.visible && !!targetWindow
    readonly property color foreground: colors ? colors.text : "#000000"
    readonly property color background: colors ? colors.inset : "#FFFFFF"
    readonly property color selected: colors ? colors.selected : "#FFFFFF"
    readonly property color accent: colors ? colors.accent : "#000000"
    readonly property bool highContrast: !!colors && Qt.colorEqual(selected, background)
    readonly property color selectionSurface: chooseSelectionSurface()

    function luminance(color) {
        function linear(channel) {
            return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * linear(color.r) + 0.7152 * linear(color.g) + 0.0722 * linear(color.b)
    }
    function contrast(first, second) {
        let a = luminance(first), b = luminance(second)
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)
    }
    function mix(first, second, amount) {
        return Qt.rgba(first.r + (second.r - first.r) * amount,
                       first.g + (second.g - first.g) * amount,
                       first.b + (second.b - first.b) * amount, 1)
    }
    function chooseSelectionSurface() {
        if (!highContrast) return selected
        // The normal high-contrast selection is a border on the background.
        // Qt's fallback picker needs a filled surface for its selected rows.
        for (let endpoint of [accent, foreground]) {
            for (let step = 0; step <= 100; ++step) {
                let candidate = mix(background, endpoint, step / 100)
                if (contrast(candidate, foreground) >= 4.5 && contrast(candidate, background) >= 3)
                    return candidate
            }
        }
        return selected
    }

    // Qt's fallback delegates pair highlightedText with palette.light, while
    // text editors pair it with palette.highlight. Keep both pairs readable
    // for the lifetime of the modal picker, then restore the window's bindings.
    property Binding lightBinding: Binding {
        target: helper.targetWindow
        property: "palette.light"
        value: helper.selectionSurface
        when: helper.active
        restoreMode: Binding.RestoreBindingOrValue
    }
    property Binding highlightBinding: Binding {
        target: helper.targetWindow
        property: "palette.highlight"
        value: helper.selectionSurface
        when: helper.active
        restoreMode: Binding.RestoreBindingOrValue
    }
    property Binding highlightedTextBinding: Binding {
        target: helper.targetWindow
        property: "palette.highlightedText"
        value: helper.foreground
        when: helper.active
        restoreMode: Binding.RestoreBindingOrValue
    }
    property Binding midBinding: Binding {
        target: helper.targetWindow
        property: "palette.mid"
        value: helper.highContrast ? helper.selectionSurface : helper.colors ? helper.colors.border : "transparent"
        when: helper.active
        restoreMode: Binding.RestoreBindingOrValue
    }
}
