pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: range
    required property var firstDescriptor
    required property var secondDescriptor
    required property var bridge
    required property var colors
    readonly property real readableCellWidth: Math.max(92, Math.ceil(metrics.averageCharacterWidth * 9 + 32))
    readonly property bool stacked: width < readableCellWidth * 2 + 8
    spacing: 4
    visible: firstDescriptor.visible !== false || secondDescriptor.visible !== false
    objectName: "app_parameter_range_" + firstDescriptor.pair

    FontMetrics { id: metrics; font: title.font }
    Basic.Label {
        id: title
        Layout.fillWidth: true
        text: range.firstDescriptor.pairLabel || "Range"
        color: range.colors.text
        wrapMode: Text.WordWrap
        Accessible.name: text
    }
    GridLayout {
        Layout.fillWidth: true
        columns: range.stacked ? 1 : 2
        columnSpacing: 8
        rowSpacing: 6
        AppParameterField {
            Layout.row: 0
            Layout.column: 0
            Layout.fillWidth: true
            Layout.preferredWidth: 1
            Layout.minimumWidth: 0
            Layout.alignment: Qt.AlignTop
            descriptor: range.firstDescriptor
            bridge: range.bridge
            colors: range.colors
            caption: descriptor.shortLabel || "Minimum"
        }
        AppParameterField {
            Layout.row: range.stacked ? 1 : 0
            Layout.column: range.stacked ? 0 : 1
            Layout.fillWidth: true
            Layout.preferredWidth: 1
            Layout.minimumWidth: 0
            Layout.alignment: Qt.AlignTop
            descriptor: range.secondDescriptor
            bridge: range.bridge
            colors: range.colors
            caption: descriptor.shortLabel || "Maximum"
        }
    }
}
