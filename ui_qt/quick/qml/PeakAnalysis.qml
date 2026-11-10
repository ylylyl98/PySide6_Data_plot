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
    minimumWidth: 760; minimumHeight: state.kind === "PL" ? 720 : 600
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
            AppButton {
                colors: window.colors; text: "Save results"
                objectName: "peak_save_results"
                enabled: !window.state.busy && window.state.canExport
                Basic.ToolTip.visible: hovered
                Basic.ToolTip.text: "Save Excel tables, source plot overlay and analysis JSON" +
                    (window.state.saveFolder ? "\n" + window.state.saveFolder : "")
                onClicked: window.peakBridge.invoke("save-results")
            }
            AppButton {
                colors: window.colors; text: "Save figure"
                objectName: "peak_action_save-figure-auto"
                enabled: !window.state.busy && window.state.hasDataset
                Basic.ToolTip.visible: hovered && !!window.state.saveFolder
                Basic.ToolTip.text: window.state.saveFolder || ""
                onClicked: window.peakBridge.invoke("save-figure-auto")
            }
            AppButton {
                colors: window.colors; text: "Apply to main app"
                objectName: "peak_action_apply-overlay"
                enabled: !window.state.busy && window.state.canApplyOverlay
                onClicked: window.peakBridge.invoke("apply-overlay")
            }
            AppButton {
                id: moreButton
                colors: window.colors; text: "More…"
                enabled: !window.state.busy
                onClicked: moreMenu.open()
                Basic.Menu {
                    id: moreMenu
                    y: moreButton.height
                    Basic.MenuItem {
                        text: "Save results as…"
                        objectName: "peak_action_export-dialog"
                        enabled: window.state.canExport && !window.state.busy
                        onTriggered: window.peakBridge.invoke("export-dialog")
                    }
                    Basic.MenuItem {
                        text: "Save figure as…"
                        objectName: "peak_action_save-figure"
                        enabled: window.state.hasDataset && !window.state.busy
                        onTriggered: window.peakBridge.invoke("save-figure")
                    }
                    Basic.MenuItem {
                        text: "Open save folder"
                        objectName: "peak_action_open-save-folder"
                        enabled: !!window.state.saveFolder && !window.state.busy
                        onTriggered: window.peakBridge.invoke("open-save-folder")
                    }
                    Basic.MenuSeparator { }
                    Repeater {
                        model: [{key: "open-workspace", label: "Open workspace…"}, {key: "save-copy", label: "Save copy…"},
                            {key: "results", label: "Results"}, {key: "task-details", label: "Task details"},
                            {key: "restart", label: "Restart"}]
                        delegate: Basic.MenuItem {
                            required property var modelData
                            text: modelData.label
                            objectName: "peak_action_" + modelData.key
                            onTriggered: window.peakBridge.invoke(modelData.key)
                        }
                    }
                    Basic.MenuSeparator { }
                    Basic.MenuItem {
                        text: "Show previous datasets"
                        objectName: "peak_show_previous_datasets"
                        checkable: true
                        checked: window.state.showHistory || false
                        enabled: window.state.historyCount > 0
                        onTriggered: window.peakBridge.setParameter("showHistory", checked)
                    }
                    Basic.MenuSeparator { }
                    Basic.MenuItem { text: "Back"; enabled: window.state.hasDataset; onTriggered: window.peakBridge.invoke("back") }
                    Basic.MenuItem { text: "Forward"; enabled: window.state.hasDataset; onTriggered: window.peakBridge.invoke("forward") }
                    Basic.MenuItem { text: "Larger peak markers"; enabled: window.state.hasDataset; onTriggered: window.peakBridge.setParameter("markerSize", (window.state.markerSize || 4) + 1) }
                    Basic.MenuItem { text: "Smaller peak markers"; enabled: window.state.hasDataset; onTriggered: window.peakBridge.setParameter("markerSize", (window.state.markerSize || 4) - 1) }
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
                        width: Math.min(320, Math.max(160, implicitWidth))
                        description: modelData.label
                        objectName: "peak_dataset_" + modelData.key
                        enabled: !window.state.busy
                        onClicked: window.peakBridge.setParameter("dataset", modelData.key)
                    }
                }
            }
        }
        Flow {
            Layout.fillWidth: true; Layout.leftMargin: 8; Layout.rightMargin: 8; Layout.bottomMargin: 5
            visible: window.state.kind === "PL"
            spacing: 6
            Repeater {
                model: [{key: "find", label: "1  Find peaks"}, {key: "branches", label: "2  Check peaks"},
                        {key: "fit", label: "3  Fit"}, {key: "results", label: "4  Results"}]
                delegate: AppButton {
                    required property var modelData
                    objectName: "peak_step_" + modelData.key
                    text: modelData.label; colors: window.colors
                    checked: window.state.controlsPage === modelData.key
                    enabled: window.state.hasDataset && !window.state.busy
                    onClicked: window.peakBridge.setParameter("controlsPage", modelData.key)
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
            ColumnLayout {
                Basic.SplitView.fillWidth: true
                Layout.minimumWidth: 0
                spacing: 2
                Flow {
                    Layout.fillWidth: true; Layout.margins: 4
                    spacing: 4
                    Repeater {
                        model: window.state.products || []
                        delegate: AppButton {
                            required property var modelData
                            colors: window.colors; text: modelData.label; checked: modelData.active
                            enabled: modelData.available && !window.state.busy
                            onClicked: window.peakBridge.setParameter("product", modelData.channel)
                        }
                    }
                    RowLayout {
                        spacing: 6
                        Basic.Label { text: "Inspect spectrum"; color: window.colors.muted }
                        Basic.SpinBox {
                            objectName: "peak_scan_row"
                            from: 0; to: window.state.rowMaximum || 0
                            value: window.state.row || 0
                            editable: true; enabled: window.state.hasDataset && !window.state.busy
                            Accessible.name: "Measured spectrum row"
                            onValueModified: window.peakBridge.setParameter("row", value)
                        }
                        Basic.Label {
                            text: (window.state.scanLabel || "Scan") + " = " + Number(window.state.scanValue || 0).toPrecision(5)
                            color: window.colors.text
                            Layout.maximumWidth: 150
                            elide: Text.ElideRight
                        }
                    }
                }
                Flow {
                    Layout.fillWidth: true
                    spacing: 2
                    Repeater {
                        model: ["home", "pan", "zoom"]
                        delegate: AppButton {
                            required property string modelData
                            colors: window.colors
                            text: modelData.charAt(0).toUpperCase() + modelData.slice(1)
                            checked: window.state.plotTool === modelData
                            Basic.ToolTip.visible: hovered
                            Basic.ToolTip.text: "Wheel: zoom · Middle-drag: pan · Home: full range"
                            enabled: window.state.hasDataset && !window.state.busy
                            onClicked: window.peakBridge.invoke(modelData)
                        }
                    }
                    Basic.CheckBox {
                        text: "Show positions"; checked: window.state.showOverlay
                        enabled: window.state.hasDataset
                        onToggled: window.peakBridge.setParameter("showOverlay", checked)
                    }
                    RowLayout {
                        spacing: 4
                        Basic.Label { text: "Size"; color: window.colors.muted }
                        Basic.SpinBox {
                            objectName: "peak_marker_size"
                            from: 1; to: 64; value: window.state.markerSize || 4
                            enabled: window.state.hasDataset
                            Accessible.name: "Position marker size"
                            Basic.ToolTip.visible: hovered
                            Basic.ToolTip.text: "Change the size of position markers on the map"
                            onValueModified: window.peakBridge.setParameter("markerSize", value)
                        }
                    }
                    RowLayout {
                        spacing: 4
                        Basic.Label { text: "Map size"; color: window.colors.muted }
                        Basic.Slider {
                            objectName: "peak_map_height"
                            implicitWidth: 100
                            from: 20; to: 80; stepSize: 1
                            value: window.state.mapHeightPercent || 69
                            enabled: window.state.hasDataset && !window.state.busy
                            Accessible.name: "Map height percentage; remaining height is for the spectrum"
                            onMoved: window.peakBridge.setParameter("mapHeightPercent", value)
                        }
                        Basic.Label { text: (window.state.mapHeightPercent || 69) + "%"; color: window.colors.text }
                        AppButton {
                            text: "Auto"; colors: window.colors; compact: true
                            description: "Restore automatic map and spectrum heights"
                            enabled: window.state.hasDataset && !window.state.busy
                            onClicked: window.peakBridge.setParameter("mapHeightPercent", "auto")
                        }
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
                    Layout.preferredHeight: visible ? Math.min(245, Math.max(100, window.height * 0.22)) : 0
                    visible: window.state.showResults
                    bridge: window.peakBridge; colors: window.colors
                }
            }
            PeakAnalysisControls {
                Basic.SplitView.preferredWidth: window.width >= 1000 ? 330 : 280
                Basic.SplitView.minimumWidth: 250
                Basic.SplitView.maximumWidth: 520
                bridge: window.peakBridge; colors: window.colors
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
        title: "Save Excel, source plot overlay and JSON"
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
