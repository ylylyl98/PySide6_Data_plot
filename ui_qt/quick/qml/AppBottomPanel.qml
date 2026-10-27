pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: panel
    required property var colors
    property bool resultsAvailable: true
    property bool logAvailable: false
    property string results: ""
    property string log: ""
    property string selectedPanel: "results"
    signal closeRequested(string panel)
    color: colors.surface
    Accessible.role: Accessible.Pane
    Accessible.name: selectedPanel === "results" ? "Analysis results" : "Run log"

    function reconcileSelection() {
        if (selectedPanel === "results" && !resultsAvailable && logAvailable) selectedPanel = "log"
        else if (selectedPanel === "log" && !logAvailable && resultsAvailable) selectedPanel = "results"
    }
    onResultsAvailableChanged: reconcileSelection()
    onLogAvailableChanged: reconcileSelection()
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        RowLayout {
            Layout.fillWidth: true
            Layout.margins: 4
            spacing: 4
            AppButton {
                objectName: "app_results_tab"
                colors: panel.colors
                text: "Analysis results"
                visible: panel.resultsAvailable
                checkable: true
                checked: panel.selectedPanel === "results"
                onClicked: panel.selectedPanel = "results"
            }
            AppButton {
                objectName: "app_log_tab"
                colors: panel.colors
                text: "Run log"
                visible: panel.logAvailable
                checkable: true
                checked: panel.selectedPanel === "log"
                onClicked: panel.selectedPanel = "log"
            }
            Item { Layout.fillWidth: true }
            AppButton {
                objectName: "app_close_panel"
                colors: panel.colors
                text: "×"
                compact: true
                description: "Hide " + (panel.selectedPanel === "results" ? "analysis results" : "run log")
                onClicked: panel.closeRequested(panel.selectedPanel)
            }
        }
        Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: panel.colors.divider }
        Basic.ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            Basic.TextArea {
                objectName: "app_panel_text"
                text: panel.selectedPanel === "results" ? panel.results : panel.log
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                color: panel.colors.text
                selectedTextColor: panel.colors.selectedText
                selectionColor: panel.colors.accent
                background: null
                padding: 12
                Accessible.name: panel.selectedPanel === "results" ? "Analysis results text" : "Run log text"
            }
        }
    }
}
