pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Layouts

Rectangle {
    id: root
    objectName: "quick_tools_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    color: colors.inset
    Controls.ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true
        ColumnLayout {
            width: parent.width
            spacing: 16
            Controls.Label { Layout.fillWidth: true; Layout.margins: 20; text: "Analysis and tools"; color: root.colors.text; font.pixelSize: 22; font.bold: true }
            GridLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 20; Layout.rightMargin: 20
                columns: width > 700 ? 2 : 1
                rowSpacing: 16; columnSpacing: 16
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: analysisColumn.implicitHeight + 32
                    color: root.colors.surface
                    border.color: root.colors.border
                    radius: 6
                    ColumnLayout {
                        id: analysisColumn
                        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
                        anchors.margins: 16
                        spacing: 10
                        Controls.Label { text: "DRR Analysis"; color: root.colors.text; font.bold: true }
                        Controls.Label { Layout.fillWidth: true; text: "Compare saved measurements, inspect candidate peaks, and calculate peak-to-peak signals in a separate workspace."; wrapMode: Text.WordWrap; color: root.colors.muted }
                        AppButton { objectName: "tools_open_drr_analysis"; colors: root.colors; text: "Open DRR Analysis…"; enabled: !root.state.drrAnalysisBusy; onClicked: root.bridge.invoke("drr-analysis") }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: organizerColumn.implicitHeight + 32
                    color: root.colors.surface
                    border.color: root.colors.border
                    radius: 6
                    ColumnLayout {
                        id: organizerColumn
                        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
                        anchors.margins: 16
                        spacing: 10
                        Controls.Label { text: "MCD Organizer"; color: root.colors.text; font.bold: true }
                        Controls.Label { Layout.fillWidth: true; text: "Group resonance energies, compare field slopes, and fit temperature dependence across saved MCD measurements."; wrapMode: Text.WordWrap; color: root.colors.muted }
                        AppButton { objectName: "tools_open_mcd_organizer"; colors: root.colors; text: "Open MCD Organizer…"; onClicked: root.bridge.invoke("mcd-organizer") }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: logColumn.implicitHeight + 32
                    color: root.colors.surface
                    border.color: root.colors.border
                    radius: 6
                    ColumnLayout {
                        id: logColumn
                        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
                        anchors.margins: 16
                        spacing: 10
                        Controls.Label { text: "Run log"; color: root.colors.text; font.bold: true }
                        Controls.Label { Layout.fillWidth: true; text: "Review loading, processing and export messages for this session."; wrapMode: Text.WordWrap; color: root.colors.muted }
                        RowLayout {
                            AppButton { colors: root.colors; text: root.state.logVisible ? "Hide log" : "Show log"; onClicked: root.bridge.invoke("toggle-log") }
                            AppButton { objectName: "tools_clear_log"; colors: root.colors; text: "Clear log"; onClicked: root.bridge.invoke("clear-log") }
                        }
                    }
                }
            }
            Controls.Label { Layout.fillWidth: true; Layout.margins: 20; text: "Peak Analysis is available from the plot toolbar or Workflow menu on PL, DRR, Compare intensity and Power intensity pages."; color: root.colors.muted; wrapMode: Text.WordWrap }
        }
    }
}
