pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Dptk 1.0
import "DialogState.js" as DialogState

Basic.ApplicationWindow {
    id: window
    required property var peakBridge
    readonly property var state: peakBridge.snapshot
    readonly property var colors: peakBridge.colors
    objectName: "peak_analysis_window"
    title: state.datasetName ? "Peak Analysis · " + state.datasetName : "Peak Analysis"
    width: 1240; height: 900
    minimumWidth: 760; minimumHeight: 600
    visible: true
    color: colors.inset
    palette.window: colors.surface
    palette.base: colors.control
    palette.text: colors.text
    palette.windowText: colors.text
    palette.buttonText: colors.text
    palette.button: colors.control
    palette.alternateBase: colors.surface
    palette.brightText: colors.text
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.mid: colors.border
    palette.midlight: colors.pressed
    palette.dark: colors.border
    palette.light: colors.control
    palette.placeholderText: colors.muted
    onClosing: function(event) {
        if (!peakBridge.snapshot.closeReady) {
            event.accepted = false
            peakBridge.requestClose()
        }
    }
    header: ColumnLayout {
        spacing: 2
        Flow {
            Layout.fillWidth: true
            Layout.margins: 6
            spacing: 4
            Repeater {
                model: [{key: "open-workspace", label: "Open workspace…"}, {key: "save-copy", label: "Save copy…"},
                    {key: "export-dialog", label: "Export CSV / JSON / PNG…"}, {key: "apply-overlay", label: "Apply to main app"},
                    {key: "results", label: "Results"}, {key: "task-details", label: "Task details"}, {key: "restart", label: "Restart"}]
                delegate: AppButton {
                    required property var modelData
                    colors: window.colors; text: modelData.label
                    objectName: "peak_action_" + modelData.key
                    enabled: !window.state.busy && (modelData.key === "export-dialog" ? window.state.canExport
                        : modelData.key === "apply-overlay" ? window.state.canApplyOverlay : true)
                    onClicked: window.peakBridge.invoke(modelData.key)
                }
            }
        }
        Basic.ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: window.state.datasetCount ? 39 : 0
            visible: window.state.datasetCount > 0
            contentHeight: 34
            Basic.ScrollBar.vertical.policy: Basic.ScrollBar.AlwaysOff
            Row {
                spacing: 4
                Repeater {
                    model: window.state.datasetChoices || []
                    delegate: AppButton {
                        required property var modelData
                        colors: window.colors; text: modelData.label; checked: modelData.active
                        objectName: "peak_dataset_" + modelData.key
                        enabled: !window.state.busy
                        onClicked: window.peakBridge.setParameter("dataset", modelData.key)
                    }
                }
            }
        }
    }
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Basic.SplitView {
            Layout.fillWidth: true; Layout.fillHeight: true
            orientation: Qt.Horizontal
            PeakAnalysisControls {
                Basic.SplitView.preferredWidth: 330
                Basic.SplitView.minimumWidth: 250
                Basic.SplitView.maximumWidth: 520
                bridge: window.peakBridge; colors: window.colors
            }
            ColumnLayout {
                Basic.SplitView.fillWidth: true
                spacing: 2
                RowLayout {
                    Layout.fillWidth: true; Layout.margins: 4
                    Repeater {
                        model: window.state.products || []
                        delegate: AppButton {
                            required property var modelData
                            colors: window.colors; text: modelData.label; checked: modelData.active
                            enabled: modelData.available && !window.state.busy
                            onClicked: window.peakBridge.setParameter("product", modelData.channel)
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Basic.Label { text: window.state.scanLabel || "Scan"; color: window.colors.muted }
                    Basic.SpinBox {
                        objectName: "peak_scan_row"
                        from: 0; to: window.state.rowMaximum || 0
                        value: window.state.row || 0
                        editable: true; enabled: window.state.hasDataset && !window.state.busy
                        Accessible.name: "Measured spectrum row"
                        onValueModified: window.peakBridge.setParameter("row", value)
                    }
                    Basic.Label { text: Number(window.state.scanValue || 0).toPrecision(5); color: window.colors.text }
                }
                Flow {
                    Layout.fillWidth: true
                    spacing: 2
                    Repeater {
                        model: ["home", "back", "forward", "pan", "zoom", "save-figure"]
                        delegate: AppButton {
                            required property string modelData
                            colors: window.colors
                            text: modelData === "save-figure" ? "Save figure…" : modelData.charAt(0).toUpperCase() + modelData.slice(1)
                            checked: window.state.plotTool === modelData
                            enabled: window.state.hasDataset && !window.state.busy
                            onClicked: window.peakBridge.invoke(modelData)
                        }
                    }
                    Basic.CheckBox {
                        text: "Show peaks"; checked: window.state.showOverlay
                        enabled: window.state.hasDataset
                        onToggled: window.peakBridge.setParameter("showOverlay", checked)
                    }
                    Basic.SpinBox {
                        from: 1; to: 64; value: window.state.markerSize || 9
                        enabled: window.state.hasDataset
                        Accessible.name: "Peak marker size"
                        onValueModified: window.peakBridge.setParameter("markerSize", value)
                    }
                }
                PlotSurface {
                    objectName: "peak_plot_surface"
                    Layout.fillWidth: true; Layout.fillHeight: true
                    backend: window.peakBridge.plotBackend
                    Basic.Label {
                        anchors.centerIn: parent
                        width: Math.min(440, parent.width - 40)
                        visible: !window.state.hasDataset
                        text: "Open Peak Analysis from a plotted PL, DRR, Compare intensity or Power intensity view, or open a saved workspace."
                        wrapMode: Text.WordWrap; horizontalAlignment: Text.AlignHCenter
                        color: window.colors.muted
                    }
                }
                Basic.Label {
                    Layout.fillWidth: true; Layout.leftMargin: 8; Layout.rightMargin: 8
                    text: window.peakBridge.cursor.text
                    color: window.colors.muted; elide: Text.ElideRight
                    HoverHandler { id: cursorHover }
                    Basic.ToolTip.visible: cursorHover.hovered
                    Basic.ToolTip.text: window.peakBridge.cursor.tooltip
                    Basic.ToolTip.delay: 500
                    Accessible.name: text
                }
                PeakAnalysisResults {
                    Layout.fillWidth: true
                    Layout.preferredHeight: visible ? 245 : 0
                    visible: window.state.showResults
                    bridge: window.peakBridge; colors: window.colors
                }
            }
        }
        RowLayout {
            Layout.fillWidth: true; Layout.margins: 7
            Basic.Label {
                objectName: "peak_status"
                Layout.fillWidth: true
                text: window.state.status || ""
                color: window.state.error ? window.colors.error : window.colors.muted
                wrapMode: Text.WordWrap
            }
            Basic.ProgressBar { visible: window.state.busy; from: 0; to: 100; value: window.state.progress || 0; Layout.preferredWidth: 100 }
            AppButton {
                text: "Cancel"; colors: window.colors
                visible: window.state.busy; enabled: !window.state.closing
                onClicked: window.peakBridge.invoke("cancel")
            }
        }
    }
    Basic.Popup {
        x: (window.width - width) / 2; y: (window.height - height) / 2
        width: Math.min(700, window.width - 60); height: Math.min(460, window.height - 60)
        visible: window.state.showDetails
        modal: true
        onClosed: window.peakBridge.invoke("close-details")
        ColumnLayout {
            anchors.fill: parent
            Basic.Label { text: "Task details"; color: window.colors.text; font.bold: true }
            Basic.ScrollView {
                Layout.fillWidth: true; Layout.fillHeight: true
                Basic.TextArea { text: window.state.taskDetails || ""; readOnly: true; wrapMode: Text.Wrap; selectByMouse: true }
            }
            AppButton { text: "Close"; colors: window.colors; onClicked: window.peakBridge.invoke("close-details") }
        }
    }
    AppFileDialog {
        colors: window.colors
        parentWindow: window
        id: fileDialog
        objectName: "peak_file_dialog"
        options: FileDialog.DontUseNativeDialog
        property string action: ""
        nameFilters: action === "save-figure" ? ["PNG image (*.png)"] : ["Peak workspace (*.npz)"]
        fileMode: action === "open-workspace" ? FileDialog.OpenFile : FileDialog.SaveFile
        defaultSuffix: action === "save-figure" ? "png" : "npz"
        onAccepted: window.peakBridge.openFiles(action === "open-workspace" ? "workspace" : action, [selectedFile])
    }
    AppFolderDialog {
        colors: window.colors
        parentWindow: window
        id: exportDialog
        objectName: "peak_export_dialog"
        options: FolderDialog.DontUseNativeDialog
        title: "Export complete analysis package"
        onAccepted: window.peakBridge.exportTo(selectedFolder.toString())
    }
    Connections {
        target: window.peakBridge
        function onDialogRequested(action, folder) {
            if (action === "export-dialog") {
                DialogState.prepareFolder(exportDialog, folder)
                exportDialog.open()
            } else {
                fileDialog.action = action
                DialogState.prepareFile(fileDialog, window.state.sessionFolderUrl, "")
                fileDialog.open()
            }
        }
    }
}
