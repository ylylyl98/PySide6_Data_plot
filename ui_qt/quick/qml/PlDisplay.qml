pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window

Rectangle {
    id: root
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    readonly property string product: state.scale || "linear"
    objectName: "pl_display"
    implicitWidth: 320
    implicitHeight: 680
    color: colors.surface
    Accessible.name: "PL display settings"

    QtObject {
        id: viewBridge
        readonly property var snapshot: root.state
        readonly property string error: root.bridge.error
        signal deactivating()
        function setNumber(key, text) { return root.bridge.setNumber(key, text) }
        function stepNumber(key, direction, draft) {
            return draft === undefined ? root.bridge.stepNumber(key, direction) : root.bridge.stepNumber(key, direction, draft)
        }
        function setChoice(key, index) { return root.bridge.setChoice(key, index) }
        function setFlag(key, checked) { return root.bridge.setFlag(key, checked) }
        function invoke(action) { return root.bridge.invoke(action) }
        function setEditing(key, editing) { root.bridge.setEditing(key, editing) }
    }
    Connections {
        target: root.bridge
        function onDeactivating() { viewBridge.deactivating() }
    }

    // Validate drafts before changing the bank; a rejected edit remains visible.
    function commitEdits() {
        function visit(item) {
            if (item.objectName && item.objectName.indexOf("drr_number_") === 0) {
                if (!item.commit()) return false
            }
            let children = item.children || []
            for (let i = 0; i < children.length; ++i)
                if (!visit(children[i])) return false
            return true
        }
        return visit(form)
    }
    function selectScale(index) {
        if (!commitEdits()) return false
        return bridge.setChoice("scale", index)
    }
    function ensureFocusVisible() {
        let window = root.Window.window
        let item = window ? window.activeFocusItem : null
        let ancestor = item
        while (ancestor && ancestor !== form) ancestor = ancestor.parent
        if (!item || ancestor !== form) return
        let position = item.mapToItem(form, 0, 0)
        let flick = scroll.contentItem as Flickable
        if (position.y < flick.contentY + 8) flick.contentY = Math.max(0, position.y - 8)
        else if (position.y + item.height + 8 > flick.contentY + flick.height)
            flick.contentY = Math.min(Math.max(0, flick.contentHeight - flick.height), position.y + item.height + 8 - flick.height)
    }
    Connections {
        target: root.Window.window
        function onActiveFocusItemChanged() { Qt.callLater(root.ensureFocusVisible) }
    }
    component SectionLabel: Text {
        Layout.fillWidth: true
        color: root.colors.text
        font.weight: Font.DemiBold
        wrapMode: Text.Wrap
    }
    component ScaleTab: Basic.TabButton {
        required property int scaleIndex
        Layout.fillWidth: true
        focusPolicy: Qt.TabFocus
        autoExclusive: false
        checked: root.product === (scaleIndex === 0 ? "linear" : "log")
        onClicked: {
            root.selectScale(scaleIndex)
            checked = Qt.binding(function() { return root.product === (scaleIndex === 0 ? "linear" : "log") })
        }
        implicitHeight: 34
        contentItem: Text {
            text: parent.text
            color: parent.checked ? root.colors.accentText : root.colors.text
            font.weight: parent.checked ? Font.DemiBold : Font.Normal
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            radius: 5
            color: parent.checked ? root.colors.control : root.colors.inset
            border.color: root.colors.border
        }
    }
    Basic.ScrollView {
        id: scroll
        objectName: "pl_display_scroll"
        anchors.fill: parent
        padding: 12
        clip: true
        contentWidth: availableWidth
        Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
        ColumnLayout {
            id: form
            width: scroll.availableWidth
            spacing: 12
            SectionLabel { text: "Axes" }
            Text { Layout.fillWidth: true; text: "Auto adjusts only limits without Fix."; color: root.colors.muted; wrapMode: Text.Wrap }
            RangeEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; title: "X"; minimumKey: "axis.xmin"; maximumKey: "axis.xmax"; autoAction: "auto.x" }
            RangeEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; title: "Y"; minimumKey: "axis.ymin"; maximumKey: "axis.ymax"; autoAction: "auto.y" }
            Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: root.colors.divider }
            SectionLabel { text: "Regions" }
            ChoiceEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "regionCount" }
            Repeater {
                model: root.state.regionCount > 1 ? root.state.regionCount - 1 : 0
                delegate: ColumnLayout {
                    required property int index
                    Layout.fillWidth: true
                    spacing: 4
                    RowLayout {
                        Layout.fillWidth: true
                        Text { Layout.fillWidth: true; text: "Boundary " + (index + 1) + " (eV)"; color: root.colors.muted }
                        FlagEditor { bridge: viewBridge; colors: root.colors; fieldKey: "fix.boundary.x" + index; compact: true }
                    }
                    NumberEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "boundary.x" + index }
                }
            }
            FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "boundary.show"; visible: root.state.regionCount > 1 }
            Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: root.colors.divider }
            SectionLabel { text: "Color scale" }
            RowLayout {
                Layout.fillWidth: true
                spacing: 3
                ScaleTab { objectName: "pl_tab_linear"; text: "Linear"; scaleIndex: 0 }
                ScaleTab { objectName: "pl_tab_log"; text: "Log"; scaleIndex: 1 }
            }
            FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "sideBySide"; label: "Side by side" }
            Text { Layout.fillWidth: true; text: "Color map"; color: root.colors.muted }
            ChoiceEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "cmap" }
            RangeEditor {
                Layout.fillWidth: true
                visible: root.state.regionCount === 1
                bridge: viewBridge; colors: root.colors; title: root.product === "linear" ? "Linear color" : "Log color"
                minimumKey: root.product + ".vmin"; maximumKey: root.product + ".vmax"; autoAction: "auto." + root.product
            }
            Repeater {
                model: root.state.regionCount === 3 ? ["left", "middle", "right"] : root.state.regionCount === 2 ? ["left", "right"] : []
                delegate: RangeEditor {
                    required property string modelData
                    Layout.fillWidth: true
                    bridge: viewBridge; colors: root.colors
                    title: modelData.charAt(0).toUpperCase() + modelData.slice(1) + " color"
                    minimumKey: root.product + "." + modelData + "_vmin"
                    maximumKey: root.product + "." + modelData + "_vmax"
                    autoAction: "auto." + root.product + "." + modelData
                }
            }
            FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "clip"; label: "Clip outliers" }
            Text { Layout.fillWidth: true; text: root.bridge.error; visible: text.length > 0; color: root.colors.error; wrapMode: Text.Wrap }
            Text { Layout.fillWidth: true; text: (root.state.texts.memoryStatus || {}).text || ""; color: root.colors.muted; wrapMode: Text.Wrap }
        }
    }
}
