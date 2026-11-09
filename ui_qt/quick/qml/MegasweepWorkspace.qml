pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    objectName: "quick_megasweep_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    property bool backgroundOpen: false
    property string lastMode: state.mode || ""
    onStateChanged: { if ((state.mode || "") !== lastMode) { backgroundOpen = false; lastMode = state.mode || "" } }
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "Megasweep plot and background references"
    onColorsChanged: if (bridge) bridge.setPalette(colors)
    Component.onCompleted: if (bridge) bridge.setPalette(colors)

    MegasweepMovieDialog { id: movieDialog; bridge: page.bridge; colors: page.colors }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Flow {
            Layout.fillWidth: true
            Layout.margins: 7
            spacing: 4
            AppButton { objectName: "megasweep_select_files"; colors: page.colors; text: "Open file…"; enabled: !page.state.busy; onClicked: page.bridge.invoke("select-files") }
            AppButton { objectName: "megasweep_update"; colors: page.colors; text: "Update"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("update") }
            AppButton { objectName: "megasweep_background"; colors: page.colors; text: "Background references"; visible: page.state.mode === "Reflection"; checkable: true; checked: page.backgroundOpen; onClicked: { page.backgroundOpen = !page.backgroundOpen; if (page.backgroundOpen) page.bridge.invoke("choose-background") } }
            AppButton { objectName: "megasweep_return_to_map"; colors: page.colors; text: "Return to map"; visible: page.state.mode === "Reflection" && page.state.reflectionPreviewVisible === true; enabled: !page.state.busy; onClicked: page.bridge.invoke("return-to-map") }
            AppButton { objectName: "megasweep_reflection_preview"; colors: page.colors; text: "Preview reference spectra"; visible: page.state.mode === "Reflection" && page.state.reflectionPreviewVisible !== true; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("reflection-preview") }
            AppButton { objectName: "megasweep_slice_previous"; colors: page.colors; text: "Previous slice"; visible: page.state.values && page.state.values.view === "slice"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("slice-previous") }
            AppButton { objectName: "megasweep_slice_next"; colors: page.colors; text: "Next slice"; visible: page.state.values && page.state.values.view === "slice"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("slice-next") }
            AppButton { objectName: "megasweep_export"; colors: page.colors; text: "Export CSV / PNG…"; enabled: !page.state.busy && page.state.canExport; onClicked: page.bridge.invoke("export") }
            AppButton { objectName: "megasweep_export_batch"; colors: page.colors; text: "Batch export…"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("export-batch") }
            AppButton { objectName: "megasweep_export_video"; colors: page.colors; text: "Export video…"; visible: page.state.mode !== "Transport"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: movieDialog.open() }
            AppButton { objectName: "megasweep_cancel"; colors: page.colors; text: "Cancel"; visible: page.state.busy; onClicked: page.bridge.invoke("cancel") }
        }
        Basic.SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Vertical
            handle: Rectangle { implicitHeight: 5; color: Basic.SplitHandle.pressed ? page.colors.accent : page.colors.border }
            Item {
                Basic.SplitView.fillHeight: true
                Basic.SplitView.minimumHeight: 160
                PlotSurface { objectName: "megasweep_plot_surface"; anchors.fill: parent; backend: page.bridge ? page.bridge.plotBackend : null }
                Text {
                    objectName: "megasweep_workspace_message"
                    anchors.centerIn: parent
                    width: Math.max(80, Math.min(520, parent.width - 40))
                    visible: !page.state.sourceFile || (!!page.state.error && !page.state.canExport)
                    text: page.state.error || (page.state.busy ? "Loading Megasweep data…" : "Open a file to begin.")
                    color: page.state.error ? page.colors.error : page.colors.muted
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                }
            }
            ColumnLayout {
                objectName: "megasweep_background_browser"
                visible: page.state.mode === "Reflection" && page.backgroundOpen
                Basic.SplitView.preferredHeight: visible ? 300 : 0
                Basic.SplitView.minimumHeight: visible ? 180 : 0
                Basic.SplitView.maximumHeight: visible ? 600 : 0
                BackgroundFileBrowser { Layout.fillWidth: true; Layout.fillHeight: true; bridge: page.bridge ? page.bridge.backgroundSelectionBridge : null; colors: page.colors; chosenTitle: "Selected backgrounds" }
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: 8
            spacing: 3
            Text { objectName: "megasweep_source"; Layout.fillWidth: true; text: page.state.sourceFile || page.state.sourcePath || ""; color: page.colors.text; elide: Text.ElideMiddle; font.pixelSize: 12 }
            Text { objectName: "megasweep_summary"; Layout.fillWidth: true; text: page.state.summary || ""; visible: text.length > 0; color: page.colors.text; wrapMode: Text.Wrap; font.pixelSize: 12 }
            Text { objectName: "megasweep_preview_notice"; Layout.fillWidth: true; visible: page.state.reflectionPreviewVisible === true; text: "Reference spectra preview. Return to the map to export the analysis."; color: page.colors.muted; wrapMode: Text.Wrap; font.pixelSize: 11 }
            Text { objectName: "megasweep_status"; Layout.fillWidth: true; text: page.state.error || page.state.status || ""; color: page.state.error ? page.colors.error : page.colors.muted; wrapMode: Text.Wrap; font.pixelSize: 11; Accessible.name: text }
            Basic.ProgressBar { Layout.fillWidth: true; visible: page.state.busy === true; indeterminate: true }
        }
    }
}
