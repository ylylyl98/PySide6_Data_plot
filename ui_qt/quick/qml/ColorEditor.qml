import QtQuick
import QtQuick.Layouts

ColumnLayout {
    id: control
    required property var bridge
    required property var colors
    required property string product
    spacing: 12

    objectName: "drr_color_editor_" + product

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 5
        Text {
            text: "Color map"
            font.pointSize: 8
            color: control.colors.muted
        }
        ChoiceEditor {
            Layout.fillWidth: true
            bridge: control.bridge
            colors: control.colors
            fieldKey: control.product + ".cmap"
        }
    }
    RangeEditor {
        Layout.fillWidth: true
        visible: control.bridge.snapshot.regionCount === 1
        bridge: control.bridge
        colors: control.colors
        title: "Color"
        minimumKey: control.product + ".vmin"
        maximumKey: control.product + ".vmax"
        autoAction: "auto." + control.product
        showFix: control.product === "raw"
    }
    RangeEditor {
        Layout.fillWidth: true
        visible: control.bridge.snapshot.regionCount > 1
        bridge: control.bridge
        colors: control.colors
        title: "Left region"
        minimumKey: control.product + ".left_vmin"
        maximumKey: control.product + ".left_vmax"
        autoAction: "auto." + control.product + ".left"
    }
    RangeEditor {
        Layout.fillWidth: true
        visible: control.bridge.snapshot.regionCount === 3
        bridge: control.bridge
        colors: control.colors
        title: "Middle region"
        minimumKey: control.product + ".middle_vmin"
        maximumKey: control.product + ".middle_vmax"
        autoAction: "auto." + control.product + ".middle"
    }
    RangeEditor {
        Layout.fillWidth: true
        visible: control.bridge.snapshot.regionCount > 1
        bridge: control.bridge
        colors: control.colors
        title: "Right region"
        minimumKey: control.product + ".right_vmin"
        maximumKey: control.product + ".right_vmax"
        autoAction: "auto." + control.product + ".right"
    }
}
