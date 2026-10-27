pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Control {
    id: pane
    objectName: "quick_shg_parameters"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    readonly property var modeField: {
        for (let group of (state.parameterGroups || []))
            for (let field of (group.fields || []))
                if (field.key === "mode") return field
        return ({})
    }
    implicitWidth: 320
    background: Rectangle { color: pane.colors.surface }
    Accessible.name: "SHG processing, background and angular-fit parameters"

    // Keep mode tabs outside the changing parameter schema so keyboard focus
    // survives switching between the single-source and reference/sample forms.
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Basic.TabBar {
            id: modes
            objectName: "app_parameter_mode"
            Layout.fillWidth: true
            Layout.margins: 8
            spacing: 2
            enabled: pane.modeField.enabled !== false
            currentIndex: pane.state.mode === "compare" ? 1 : 0
            Accessible.name: "SHG workflow"
            background: Rectangle { color: pane.colors.border }

            function activate(index) {
                if (enabled) pane.bridge.setParameter("mode", index === 1 ? "compare" : "single")
            }
            function focusTab(index) {
                index = Math.max(0, Math.min(1, index))
                activate(index)
                itemAt(index).forceActiveFocus(Qt.TabFocusReason)
            }
            AppTabButton {
                objectName: "shg_single"
                colors: pane.colors
                text: "Single file"
                width: (modes.availableWidth - modes.spacing) / 2
                implicitHeight: Math.max(36, contentItem.implicitHeight + 12)
                onClicked: modes.activate(0)
                Keys.onLeftPressed: modes.focusTab(0)
                Keys.onRightPressed: modes.focusTab(1)
            }
            AppTabButton {
                objectName: "shg_compare"
                colors: pane.colors
                text: "Compare / Twist"
                width: (modes.availableWidth - modes.spacing) / 2
                implicitHeight: Math.max(36, contentItem.implicitHeight + 12)
                onClicked: modes.activate(1)
                Keys.onLeftPressed: modes.focusTab(0)
                Keys.onRightPressed: modes.focusTab(1)
                Basic.ToolTip.visible: hovered
                Basic.ToolTip.text: "Compare reference A and sample B / twist angle"
                Basic.ToolTip.delay: 700
            }
        }
        AppParameterPane {
            Layout.fillWidth: true
            Layout.fillHeight: true
            bridge: pane.bridge
            colors: pane.colors
            groups: (pane.state.parameterGroups || []).map(group => Object.assign({}, group,
                {fields: (group.fields || []).filter(field => field.key !== "mode")}))
            message: pane.state.error || ""
        }
    }
}
