pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: sidebar
    property var controller: null
    required property var colors
    readonly property var state: controller ? controller.snapshot : ({})
    readonly property var live: controller ? controller.preview : ({})
    readonly property var groups: state.groups || []
    visible: groups.length > 0
    spacing: 8
    objectName: "color_range_sidebar"

    function previewFor(id) {
        return live.contextVersion === state.contextVersion ? (live.groups || {})[id] || ({}) : ({})
    }
    Text {
        Layout.fillWidth: true
        text: "Color limits" + (sidebar.state.productId === "raw" ? " · ΔR/R"
             : sidebar.state.productId === "second" ? " · Derivative"
             : sidebar.state.productId === "linear" ? " · Linear"
             : sidebar.state.productId === "log" ? " · Log" : "")
        color: sidebar.colors.text
        font.weight: Font.DemiBold
        wrapMode: Text.Wrap
    }
    Repeater {
        // An updated value must not replace the focused input or its draft.
        model: sidebar.groups.length
        delegate: ColumnLayout {
            id: region
            required property int index
            readonly property var descriptor: sidebar.groups[index] || ({})
            Layout.fillWidth: true
            spacing: 2
            Text {
                Layout.fillWidth: true
                visible: sidebar.groups.length > 1
                text: region.descriptor.title || ""
                color: sidebar.colors.text
            }
            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                ColorRangeNumber {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    controller: sidebar.controller
                    colors: sidebar.colors
                    descriptor: region.descriptor
                    endpoint: "low"
                    contextVersion: sidebar.state.contextVersion || 0
                    preview: sidebar.previewFor(region.descriptor.id)
                }
                ColorRangeNumber {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    controller: sidebar.controller
                    colors: sidebar.colors
                    descriptor: region.descriptor
                    endpoint: "high"
                    contextVersion: sidebar.state.contextVersion || 0
                    preview: sidebar.previewFor(region.descriptor.id)
                }
            }
        }
    }
    RowLayout {
        Layout.fillWidth: true
        visible: (sidebar.state.strengthOptions || []).length > 0
        Text { Layout.fillWidth: true; text: "Weak strength"; color: sidebar.colors.muted }
        Basic.ComboBox {
            objectName: "color_sidebar_weak_strength"
            enabled: sidebar.groups.some(function(group) { return group.enabled !== false })
            model: sidebar.state.strengthOptions || []
            currentIndex: sidebar.state.strengthIndex || 0
            implicitHeight: 30
            implicitWidth: 110
            palette.button: sidebar.colors.control
            palette.base: sidebar.colors.control
            palette.text: sidebar.colors.text
            palette.buttonText: sidebar.colors.text
            palette.highlight: sidebar.colors.accent
            palette.highlightedText: sidebar.colors.selectedText
            Accessible.name: "Weak feature recommendation strength"
            onActivated: sidebar.controller.setStrength(currentIndex)
        }
    }
    Text {
        Layout.fillWidth: true
        text: sidebar.state.error || "Sliders, Auto and recommendations are below the plot."
        color: sidebar.state.error ? sidebar.colors.error : sidebar.colors.muted
        wrapMode: Text.Wrap
        font.pointSize: 8
    }
}
