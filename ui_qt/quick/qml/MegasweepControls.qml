pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Control {
    id: panel
    objectName: "quick_megasweep_controls"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    implicitWidth: 320
    background: Rectangle { color: panel.colors.surface }
    Accessible.name: "Megasweep processing parameters"
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Basic.TabBar {
            id: modes
            objectName: "megasweep_modes"
            Layout.fillWidth: true
            Layout.margins: 8
            enabled: !panel.state.busy
            currentIndex: panel.state.mode === "Reflection" ? 1 : panel.state.mode === "Transport" ? 2 : 0
            background: Rectangle { color: panel.colors.border }
            AppTabButton { objectName: "megasweep_mode_pl"; colors: panel.colors; text: "PL"; onClicked: panel.bridge.setParameter("mode", "PL") }
            AppTabButton { objectName: "megasweep_mode_reflection"; colors: panel.colors; text: "Reflection"; onClicked: panel.bridge.setParameter("mode", "Reflection") }
            AppTabButton { objectName: "megasweep_mode_transport"; colors: panel.colors; text: "Transport"; onClicked: panel.bridge.setParameter("mode", "Transport") }
        }
        AppParameterPane {
            Layout.fillWidth: true
            Layout.fillHeight: true
            pinDataGroup: false
            bridge: panel.bridge
            colors: panel.colors
            groups: (panel.state.parameterGroups || []).map(group => Object.assign({}, group,
                {fields: (group.fields || []).filter(field => field.key !== "mode")}))
            message: panel.state.error || ""
        }
    }
}
