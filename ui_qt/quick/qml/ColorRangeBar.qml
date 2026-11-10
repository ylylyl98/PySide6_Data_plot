pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: bar
    property var controller: null
    required property var colors
    property real headerRightInset: 0
    property real headerMinimumHeight: 0
    readonly property var state: controller ? controller.snapshot : ({})
    readonly property var groups: state.groups || []
    readonly property bool available: groups.length > 0
    spacing: 3
    visible: available
    objectName: "color_range_bar"

    RowLayout {
        Layout.fillWidth: true
        Layout.rightMargin: bar.headerRightInset
        Layout.minimumHeight: bar.headerMinimumHeight
        spacing: 8
        Text {
            text: "C limits" + ((bar.state.products || []).length > 1 ? ""
                  : bar.state.productId === "raw" ? " · ΔR/R"
                  : bar.state.productId === "second" ? " · Derivative"
                  : bar.state.productId === "linear" ? " · Linear"
                  : bar.state.productId === "log" ? " · Log" : "")
            color: bar.colors.text; font.weight: Font.DemiBold
        }
        Basic.ComboBox {
            objectName: "color_product"
            palette.button: bar.colors.control
            palette.base: bar.colors.control
            palette.text: bar.colors.text
            palette.buttonText: bar.colors.text
            palette.highlight: bar.colors.accent
            palette.highlightedText: bar.colors.selectedText
            visible: (bar.state.products || []).length > 1
            enabled: bar.groups.some(function(group) { return group.enabled !== false })
            model: bar.state.products || []
            textRole: "label"; valueRole: "id"
            implicitHeight: 28
            implicitWidth: 115
            currentIndex: {
                let products = bar.state.products || []
                return products.findIndex(function(p) { return p.id === bar.state.productId })
            }
            Accessible.name: "Heatmap to adjust"
            onActivated: bar.controller.setProduct(currentValue)
        }
        Item { Layout.fillWidth: true }

    }
    // Keep all regions in one row. On narrow windows scroll horizontally instead
    // of shrinking numbers/targets or stacking three rows over the plot.
    Basic.ScrollView {
        id: scroll
        Layout.fillWidth: true
        Layout.preferredHeight: row.implicitHeight + (contentWidth > availableWidth ? 12 : 0)
        clip: true
        contentWidth: Math.max(availableWidth, bar.groups.length * 270 + Math.max(0, bar.groups.length - 1) * 16)
        contentHeight: row.implicitHeight
        Basic.ScrollBar.vertical.policy: Basic.ScrollBar.AlwaysOff
        Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AsNeeded
        RowLayout {
            id: row
            width: scroll.contentWidth
            spacing: 16
            Repeater {
                model: bar.groups.length
                delegate: ColorRangeGroup {
                    required property int index
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    Layout.minimumWidth: 270
                    Layout.alignment: Qt.AlignTop
                    controller: bar.controller
                    colors: bar.colors
                    descriptor: bar.groups[index] || ({})
                    contextVersion: bar.state.contextVersion || 0
                }
            }
        }
    }
    Text {
        objectName: "color_range_error"
        Layout.fillWidth: true
        text: bar.state.error || ""
        visible: text.length > 0
        color: bar.colors.error
        wrapMode: Text.Wrap
    }
}
