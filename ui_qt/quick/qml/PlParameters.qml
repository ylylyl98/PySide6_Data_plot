pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: root
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    property int page: 0
    objectName: "pl_native_parameters"
    implicitWidth: 320
    color: colors.surface
    ColumnLayout {
        anchors.fill: parent
        spacing: 4
        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: 12
            spacing: 6
            Text { text: "Data"; color: root.colors.text; font.weight: Font.DemiBold }
            Text { Layout.fillWidth: true; text: root.state.sourceSummary || "No measurement selected"; color: root.colors.muted; wrapMode: Text.Wrap }
            Flow {
                Layout.fillWidth: true
                spacing: 5
                AppButton { objectName: "app_parameter_select-files"; colors: root.colors; compact: true; text: "Select file…"; enabled: !root.state.busy; onClicked: root.bridge.invoke("select-files") }
                AppButton { objectName: "app_parameter_clear-files"; colors: root.colors; compact: true; text: "Clear"; enabled: !root.state.busy; onClicked: root.bridge.invoke("clear-files") }
                AppButton { objectName: "app_parameter_saved-results"; colors: root.colors; compact: true; text: "Saved DAT…"; enabled: !root.state.busy; onClicked: root.bridge.invoke("saved-results") }
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            AppButton { objectName: "pl_display_page"; Layout.fillWidth: true; colors: root.colors; text: "Display"; checkable: true; checked: root.page === 0; onClicked: root.page = 0 }
            AppButton { objectName: "pl_options_page"; Layout.fillWidth: true; colors: root.colors; text: "More"; checkable: true; checked: root.page === 1; onClicked: { if (display.commitEdits()) root.page = 1 } }
        }
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: root.page
            PlDisplay { id: display; bridge: root.bridge.plDisplayBridge; colors: root.colors }
            AppParameterPane {
                objectName: "pl_options"
                bridge: root.bridge; colors: root.colors
                groups: (root.state.parameterGroups || []).filter(function(group) { return ["data", "display", "axes", "plNormalization"].indexOf(group.key) < 0 })
            }
        }
    }
}
