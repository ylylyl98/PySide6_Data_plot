pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Dialogs
import QtQuick.Layouts
import Dptk 1.0
import "DialogState.js" as DialogState

Controls.ApplicationWindow {
    id: window
    objectName: "quick_application_window"
    required property var appBridge
    readonly property var state: appBridge.snapshot
    readonly property var colors: appBridge.colors
    readonly property string currentWorkflow: state.workflow || "pl"
    readonly property bool controlsVisible: state.controlsVisible !== false
    readonly property bool railVisible: state.railVisible !== false
    readonly property bool resultsVisible: state.resultsVisible === true
    readonly property bool logVisible: state.logVisible === true
    readonly property bool compact: width < 1040
    readonly property bool plotWorkflow: currentWorkflow !== "slides" && currentWorkflow !== "tools"
    property string fileDialogPurpose: "select-files"
    property string fileDialogWorkflow: "pl"
    property string folderDialogPurpose: "source-folder"
    property string folderDialogWorkflow: "pl"
    property var workflows: [
        {key: "pl", label: "PL", description: "Photoluminescence"},
        {key: "drr", label: "DRR", description: "Differential reflectance"},
        {key: "cmp", label: "Compare", description: "Compare measurement channels"},
        {key: "power", label: "Power", description: "Power dependent measurements"},
        {key: "mcd", label: "MCD", description: "Magnetic circular dichroism"},
        {key: "mcd_peak_shift", label: "MCD Peak Shift", description: "Track reflection resonance energies versus magnetic field"},
        {key: "shg", label: "SHG", description: "Second harmonic generation processing"},
        {key: "slides", label: "Slides", description: "Build a PowerPoint from processed plot PNGs"},
        {key: "tools", label: "Tools", description: "Data tools and application diagnostics"}
    ]
    width: 1440
    height: 940
    minimumWidth: 640
    minimumHeight: 520
    visible: true
    title: state.title || "DPTK Desktop"
    color: colors.inset
    font.pointSize: 9
    palette.window: colors.inset
    palette.windowText: colors.text
    palette.base: colors.control
    palette.alternateBase: colors.surface
    palette.text: colors.text
    palette.button: colors.control
    palette.buttonText: colors.text
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.mid: colors.border
    palette.midlight: colors.pressed
    palette.dark: colors.border
    palette.light: colors.control
    palette.brightText: colors.text
    palette.placeholderText: colors.muted

    function invoke(command) { appBridge.invoke(command) }
    function exportCurrent() {
        if (pngWidthEditor.visible && pngWidthEditor.enabled) {
            pngWidthEditor.commit()
            if (pngWidthEditor.invalid) return
        }
        appBridge.invoke("export")
    }
    function chooseFiles(purpose, multiple) {
        if (purpose === "output-folder") {
            chooseFolder(purpose)
            return
        }
        let requestState = appBridge.snapshot
        fileDialogPurpose = purpose
        fileDialogWorkflow = requestState.workflow
        fileDialog.fileMode = multiple ? FileDialog.OpenFiles : FileDialog.OpenFile
        fileDialog.title = purpose === "select-background" ? "Choose external DRR background files"
                         : purpose === "settings" ? "Open SHG analysis settings"
                         : purpose === "saved-results" ? "Open a saved DAT result" : "Choose measurement files"
        fileDialog.nameFilters = purpose === "saved-results" ? ["Saved result (*.dat)"]
                               : purpose === "settings" ? ["SHG analysis settings (*.json)"]
                               : purpose === "select-background" ? ["Background measurements (*.csv)"]
                               : (requestState.fileNameFilters || ["Measurement data (*.dat *.csv *.xlsx)", "All files (*)"])
        // Snapshot notifications are batched; read the current request directory now.
        DialogState.prepareFile(fileDialog, requestState.sourceFolderUrl, "")
        fileDialog.open()
    }
    function chooseFolder(purpose) {
        let requestState = appBridge.snapshot
        folderDialogPurpose = purpose || "source-folder"
        folderDialogWorkflow = requestState.workflow
        folderDialog.title = folderDialogPurpose === "output-folder" ? "Choose results folder" : "Choose a data folder"
        DialogState.prepareFolder(folderDialog, requestState.sourceFolderUrl)
        folderDialog.open()
    }
    function workflowLabel() {
        for (let workflow of workflows) if (workflow.key === currentWorkflow) return workflow.label
        return currentWorkflow
    }
    function reloadParameterPage() {
        let url = appBridge.parameterUrl
        if (String(url).length > 0) {
            parameterLoader.setSource(url, {bridge: appBridge.workflowBridge, colors: colors})
        } else parameterLoader.source = ""
    }
    function reloadWorkspacePage() {
        let url = appBridge.workspaceUrl
        if (String(url).length > 0) {
            workspaceLoader.setSource(url, {bridge: appBridge.workflowBridge, colors: colors})
        } else workspaceLoader.source = ""
    }
    function reloadPages() {
        reloadParameterPage()
        reloadWorkspacePage()
    }
    function reloadUpdater() {
        if (appBridge.updateBridge)
            updateLoader.setSource(Qt.resolvedUrl("UpdateDialog.qml"), {updateBridge: appBridge.updateBridge, colors: colors})
    }
    onColorsChanged: {
        if (parameterLoader.item) parameterLoader.item.colors = colors
        if (workspaceLoader.item) workspaceLoader.item.colors = colors
        if (backgroundLoader.item) backgroundLoader.item.colors = colors
        if (updateLoader.item) updateLoader.item.colors = colors
        if (catalogLoader.item) catalogLoader.item.colors = colors
    }
    Component.onCompleted: {
        reloadParameterPage()
        reloadWorkspacePage()
        reloadUpdater()
    }
    Connections {
        target: window.appBridge
        ignoreUnknownSignals: true
        function onUpdateBridgeChanged() {
            window.reloadUpdater()
        }
        function onParameterUrlChanged() { Qt.callLater(window.reloadPages) }
        function onWorkspaceUrlChanged() { Qt.callLater(window.reloadPages) }
        function onWorkflowBridgeChanged() { Qt.callLater(window.reloadPages) }
        function onFileDialogRequested(purpose, multiple) { window.chooseFiles(purpose, multiple) }
        function onSourceCatalogRequested() {
            if (!catalogLoader.item)
                catalogLoader.setSource(Qt.resolvedUrl("SourceCatalog.qml"), {bridge: window.appBridge.sourceCatalogBridge, colors: window.colors})
            if (catalogLoader.item) catalogLoader.item.open()
        }
        function onBackgroundDialogRequested() {
            backgroundLoader.setSource(Qt.resolvedUrl("DrrBackgroundSettings.qml"), {bridge: window.appBridge.drrBackgroundBridge, colors: window.colors})
            backgroundWindow.show()
            backgroundWindow.raise()
        }
        function onBackgroundDialogClosed() {
            backgroundWindow.hide()
            backgroundLoader.source = ""
        }
        function onBackgroundFileDialogRequested(role, multiple) { backgroundWindow.choose(role, multiple, "") }
        function onBackgroundSaveDialogRequested(role, suggested) { backgroundWindow.choose(role, false, suggested) }
    }

    Loader { id: updateLoader; anchors.fill: parent }
    Loader { id: catalogLoader; anchors.fill: parent }

    Controls.Action {
        id: openFileAction
        objectName: "app_open_file_action"
        text: qsTr("Open &File…")
        shortcut: StandardKey.Open
        enabled: window.plotWorkflow
        onTriggered: window.invoke("select-files")
    }
    Controls.Action { id: openFolderAction; text: qsTr("Open F&older…"); shortcut: "Ctrl+Shift+O"; onTriggered: window.chooseFolder("source-folder") }
    Controls.Action { id: reloadAction; text: qsTr("&Reload"); shortcut: "Ctrl+R"; enabled: window.plotWorkflow; onTriggered: window.invoke("reload") }
    Controls.Action { id: updateAction; text: qsTr("&Update now"); shortcut: "F5"; enabled: window.plotWorkflow; onTriggered: window.invoke("update") }
    Controls.Action { id: exportAction; text: window.state.exportLabel || qsTr("Save &PNG + DAT"); shortcut: "Ctrl+Shift+S"; enabled: window.state.canExport !== false && !window.state.busy; onTriggered: window.exportCurrent() }
    Controls.Action { id: controlsAction; text: qsTr("Show &Controls Sidebar"); shortcut: "Ctrl+B"; checkable: true; checked: window.controlsVisible; onTriggered: window.appBridge.setControlsVisible(!window.controlsVisible) }
    Controls.Action { id: railAction; text: qsTr("Show &Workflow Navigation"); shortcut: "Ctrl+Shift+B"; checkable: true; checked: window.railVisible; onTriggered: window.appBridge.setRailVisible(!window.railVisible) }
    Controls.Action { id: resultsAction; text: qsTr("Show Analysis &Results"); shortcut: "Ctrl+J"; checkable: true; checked: window.resultsVisible; onTriggered: { bottomPanel.selectedPanel = "results"; window.invoke("toggle-results") } }
    Controls.Action { id: logAction; text: qsTr("Show Run &Log"); shortcut: "Ctrl+Shift+J"; checkable: true; checked: window.logVisible; onTriggered: { bottomPanel.selectedPanel = "log"; window.invoke("toggle-log") } }
    Controls.Action { id: homeAction; text: qsTr("Reset plot view"); shortcut: "Alt+Home"; onTriggered: window.invoke("home") }
    Controls.Action { id: backAction; text: qsTr("Previous plot view"); shortcut: "Alt+Left"; onTriggered: window.invoke("back") }
    Controls.Action { id: forwardAction; text: qsTr("Next plot view"); shortcut: "Alt+Right"; onTriggered: window.invoke("forward") }
    Controls.Action { id: panAction; text: qsTr("Pan"); checkable: true; checked: window.state.plotTool === "pan"; onTriggered: window.invoke("pan") }
    Controls.Action { id: zoomAction; text: qsTr("Zoom"); checkable: true; checked: window.state.plotTool === "zoom"; onTriggered: window.invoke("zoom") }
    Controls.Action { id: peakAnalysisAction; text: qsTr("Peak Analysis…"); enabled: window.state.canAnalyzePeaks === true; onTriggered: window.invoke("peak-analysis") }
    Controls.Action { id: peakOverlayAction; text: qsTr("Show peak overlay"); enabled: window.state.hasPeakOverlay === true; checkable: true; checked: window.state.peakOverlayVisible !== false; onTriggered: window.invoke("toggle-peak-overlay") }

    AppFileDialog {
        colors: window.colors
        parentWindow: window
        id: fileDialog
        // The Windows native helper can crash when cancelled during opening.
        // The Quick picker keeps the same roles and supports this lifecycle.
        options: FileDialog.DontUseNativeDialog
        objectName: "app_file_dialog"
        title: "Choose a measurement file"
        fileMode: FileDialog.OpenFile
        nameFilters: window.state.fileNameFilters || ["Measurement data (*.dat *.txt *.csv *.asc)", "All files (*)"]
        onAccepted: {
            window.appBridge.acceptFiles(window.fileDialogWorkflow, window.fileDialogPurpose, selectedFiles)
        }
    }
    AppFolderDialog {
        colors: window.colors
        parentWindow: window
        id: folderDialog
        options: FolderDialog.DontUseNativeDialog
        objectName: "app_folder_dialog"
        title: "Choose a data folder"
        onAccepted: window.appBridge.acceptFolder(window.folderDialogWorkflow, window.folderDialogPurpose, selectedFolder)
    }

    Controls.ApplicationWindow {
        id: backgroundWindow
        objectName: "drr_background_window"
        title: "DRR background settings"
        width: 1060
        height: 760
        minimumWidth: 700
        minimumHeight: 560
        color: window.colors.surface
        palette: window.palette
        font: window.font
        transientParent: window
        modality: Qt.WindowModal
        flags: Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowTitleHint
        property string fileRole: "background"
        property int fileToken: 0
        function choose(role, multiple, suggested) {
            fileRole = role
            fileToken = window.appBridge.backgroundDialogToken
            backgroundFileDialog.fileMode = suggested ? FileDialog.SaveFile : multiple ? FileDialog.OpenFiles : FileDialog.OpenFile
            backgroundFileDialog.title = suggested ? "Save composite background" : role.startsWith("catalog:") ? "Browse reference files" : "Choose " + role + " files"
            DialogState.prepareFile(backgroundFileDialog, window.appBridge.snapshot.sourceFolderUrl,
                                    suggested ? window.appBridge.localFileUrl(suggested) : "")
            backgroundFileDialog.open()
        }
        Loader { id: backgroundLoader; anchors.fill: parent }
        AppFileDialog {
            colors: window.colors
            parentWindow: backgroundWindow
            id: backgroundFileDialog
            options: FileDialog.DontUseNativeDialog
            objectName: "drr_background_file_dialog"
            nameFilters: ["Background and measurement data (*.csv)"]
            defaultSuffix: "csv"
            onAccepted: window.appBridge.acceptDrrBackgroundFiles(backgroundWindow.fileToken, backgroundWindow.fileRole, selectedFiles)
        }
        onClosing: window.appBridge.cancelDrrBackground()
    }

    menuBar: Controls.MenuBar {
        objectName: "app_menu_bar"
        Controls.Menu {
            objectName: "app_file_menu"
            title: qsTr("&File")
            Controls.MenuItem { action: openFileAction }
            Controls.MenuItem { action: openFolderAction }
            Controls.Menu {
                title: qsTr("Recent folders")
                enabled: (window.state.recentFolders || []).length > 0
                Repeater {
                    model: window.state.recentFolders || []
                    delegate: Controls.MenuItem {
                        required property string modelData
                        text: modelData
                        onTriggered: window.appBridge.setSourcePath(modelData)
                    }
                }
            }
            Controls.MenuSeparator {}
            Controls.MenuItem { action: reloadAction }
            Controls.MenuItem { action: updateAction }
            Controls.MenuItem { action: exportAction }
            Controls.MenuSeparator {}
            Controls.MenuItem { text: qsTr("E&xit"); onTriggered: window.close() }
        }
        Controls.Menu {
            objectName: "app_view_menu"
            title: qsTr("&View")
            Controls.MenuItem { action: controlsAction }
            Controls.MenuItem { action: railAction }
            Controls.MenuItem { action: resultsAction }
            Controls.MenuItem { action: logAction }
            Controls.MenuSeparator {}
            Controls.MenuItem { action: homeAction }
            Controls.MenuItem { action: backAction }
            Controls.MenuItem { action: forwardAction }
            Controls.MenuItem { action: panAction }
            Controls.MenuItem { action: zoomAction }
            Controls.MenuSeparator {}
            Controls.Menu {
                title: qsTr("&Appearance")
                Controls.MenuItem { text: qsTr("Use System Theme"); checkable: true; checked: window.state.appearance === "system"; onTriggered: window.invoke("theme-system") }
                Controls.MenuItem { text: qsTr("Light"); checkable: true; checked: window.state.appearance === "light"; onTriggered: window.invoke("theme-light") }
                Controls.MenuItem { text: qsTr("Dark"); checkable: true; checked: window.state.appearance === "dark"; onTriggered: window.invoke("theme-dark") }
                Controls.MenuItem { text: qsTr("High Contrast"); checkable: true; checked: window.state.appearance === "high-contrast"; onTriggered: window.invoke("theme-high-contrast") }
            }
        }
        Controls.Menu {
            objectName: "app_workflow_menu"
            title: qsTr("&Workflow")
            Controls.MenuItem { action: peakAnalysisAction }
            Controls.MenuItem { action: peakOverlayAction }
            Controls.MenuItem { text: "DRR Analysis…"; enabled: !window.state.drrAnalysisBusy; onTriggered: window.invoke("drr-analysis") }
            Controls.MenuItem { text: "Add current DRR to Analysis"; enabled: window.state.canAddDrrAnalysis === true && !window.state.drrAnalysisBusy; onTriggered: window.invoke("drr-analysis-current") }
            Controls.MenuItem { text: "MCD Organizer…"; onTriggered: window.invoke("mcd-organizer") }
            Controls.MenuSeparator {}
            Repeater {
                model: window.workflows
                delegate: Controls.MenuItem {
                    required property var modelData
                    text: modelData.label
                    checkable: true
                    checked: window.currentWorkflow === modelData.key
                    onTriggered: window.appBridge.selectWorkflow(modelData.key)
                }
            }
        }
        Controls.Menu {
            title: qsTr("&Data Maintenance")
            Controls.MenuItem { text: qsTr("Rebuild Source Catalog"); onTriggered: window.invoke("rebuild-catalog") }
            Controls.MenuItem { text: qsTr("Move Exported Sources"); enabled: window.state.canMoveExportedSources === true && !window.state.busy; onTriggered: window.invoke("move-exported-sources") }
            Controls.MenuItem {
                text: qsTr("Clean verified source copies after successful export")
                enabled: window.state.maintenanceSupported === true && !window.state.maintenanceBusy
                checkable: true
                checked: window.state.cleanVerifiedSources === true
                onTriggered: window.appBridge.setParameter("cleanVerifiedSources", !window.state.cleanVerifiedSources)
            }
        }
        Controls.Menu {
            title: qsTr("&Help")
            Controls.MenuItem { text: qsTr("Check for Updates…"); onTriggered: window.invoke("check-updates") }
            Controls.MenuItem {
                text: qsTr("Check for Updates Automatically")
                checkable: true
                checked: window.state.autoUpdateCheck === true
                onTriggered: window.appBridge.setParameter("autoUpdateCheck", !window.state.autoUpdateCheck)
            }
            Controls.MenuItem { text: qsTr("About"); onTriggered: window.invoke("about") }
        }
    }

    header: Basic.ToolBar {
        id: sourceToolbar
        objectName: "app_source_toolbar"
        padding: 6
        background: Rectangle { color: window.colors.control; border.color: window.colors.divider }
        RowLayout {
            anchors.fill: parent
            spacing: 5
            Text { text: "Source:"; color: window.colors.muted; visible: window.width >= 800 }
            Basic.TextField {
                id: sourcePath
                objectName: "app_source_path"
                Layout.fillWidth: true
                Layout.minimumWidth: 50
                implicitHeight: 30
                text: window.state.sourcePath || ""
                placeholderText: "Choose a data folder or file"
                selectByMouse: true
                Accessible.name: "Data source path"
                Accessible.description: "Enter a data folder or use Folder and File to browse."
                onAccepted: window.appBridge.setSourcePath(text)
                DropArea {
                    anchors.fill: parent
                    onEntered: function(drag) { drag.accepted = drag.hasUrls }
                    onDropped: function(drop) { drop.accepted = window.appBridge.dropSources(drop.urls) }
                }
            }
            AppButton { objectName: "app_browse_folder"; colors: window.colors; text: "Folder…"; description: "Choose a data folder (Ctrl+Shift+O)"; onClicked: window.chooseFolder("source-folder") }
            AppButton { objectName: "app_open_file"; colors: window.colors; text: "File…"; description: "Open a measurement file (Ctrl+O)"; enabled: openFileAction.enabled; onClicked: openFileAction.trigger() }
            AppButton { objectName: "app_refresh_sources"; colors: window.colors; text: "Refresh"; description: "Refresh the source catalog"; visible: !window.compact; onClicked: window.invoke("refresh") }
            Rectangle { implicitWidth: 1; implicitHeight: 22; color: window.colors.divider; visible: !window.compact }
            AppButton { objectName: "app_reload"; colors: window.colors; text: "Reload"; description: "Reload data for this workflow (Ctrl+R)"; visible: !window.compact; enabled: reloadAction.enabled; onClicked: reloadAction.trigger() }
            AppButton { objectName: "app_update"; colors: window.colors; text: "Update now"; description: "Update the current plot (F5)"; visible: !window.compact; enabled: updateAction.enabled; onClicked: updateAction.trigger() }
            AppComboBox {
                colors: window.colors
                objectName: "app_png_layout"
                visible: window.state.pngExportSupported === true
                enabled: !window.state.busy
                Layout.preferredWidth: 112
                implicitHeight: 30
                model: ["PNG 1.0×", "Wide PNG"]
                currentIndex: window.state.pngLayout === "wide" ? 1 : 0
                onActivated: window.appBridge.setParameter("export.pngLayout", currentIndex === 1 ? "wide" : "original")
                Accessible.name: "PNG export layout"
                Basic.ToolTip.visible: hovered
                Basic.ToolTip.text: window.currentWorkflow === "drr"
                    ? "Wide PNG saves the original 1.0× and the selected wider image for each product, sharing the DAT."
                    : "Choose the exported PNG width; the on-screen plot and DAT stay unchanged."
            }
            AppNumberField {
                id: pngWidthEditor
                objectName: "app_png_width"
                visible: window.state.pngExportSupported === true
                Layout.preferredWidth: 92
                bridge: window.appBridge
                colors: window.colors
                descriptor: ({key: "export.pngWidth", label: "PNG width multiplier (×)",
                    value: window.state.pngWidth === undefined ? 2 : window.state.pngWidth,
                    min: 1, max: 3, step: 0.25,
                    enabled: window.state.pngLayout === "wide" && !window.state.busy,
                    description: "1–3× width; PNG height, text size and data ranges stay unchanged."})
            }
            Text { text: "×"; color: window.colors.muted; visible: window.state.pngExportSupported === true }
            AppButton { objectName: "app_export"; colors: window.colors; text: window.compact ? "Export" : window.state.exportLabel || "Save PNG + DAT"; description: "Export the active workflow (Ctrl+Shift+S)"; enabled: exportAction.enabled; primary: true; onClicked: exportAction.trigger() }
            AppButton { colors: window.colors; text: "More"; visible: window.compact; description: "More source and plot commands"; onClicked: sourceOverflow.open() }
            Controls.Menu {
                id: sourceOverflow
                y: sourceToolbar.height
                Controls.MenuItem { text: "Refresh source catalog"; onTriggered: window.invoke("refresh") }
                Controls.MenuItem { action: reloadAction }
                Controls.MenuItem { action: updateAction }
                Controls.MenuItem { action: exportAction }
            }
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0
        AppWorkflowRail {
            objectName: "app_workflow_rail"
            Layout.preferredWidth: 104
            Layout.fillHeight: true
            visible: window.railVisible
            colors: window.colors
            workflows: window.workflows
            currentWorkflow: window.currentWorkflow
            onSelected: function(key) { window.appBridge.selectWorkflow(key) }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0
            Rectangle {
                objectName: "app_workspace_header"
                Layout.fillWidth: true
                implicitHeight: 40
                color: window.colors.surface
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 3
                    Text {
                        objectName: "app_workspace_title"
                        Layout.preferredWidth: window.compact ? 0 : 180
                        Layout.maximumWidth: 220
                        visible: !window.compact
                        text: window.state.workspaceTitle || window.workflowLabel()
                        color: window.colors.text
                        font.weight: Font.DemiBold
                        elide: Text.ElideRight
                        Accessible.name: text
                    }
                    AppButton { objectName: "app_plot_home"; visible: window.plotWorkflow; colors: window.colors; text: "Home"; compact: true; description: "Reset plot view (Alt+Home)"; onClicked: homeAction.trigger() }
                    AppButton { objectName: "app_plot_back"; visible: window.plotWorkflow; colors: window.colors; text: "‹"; compact: true; description: "Previous plot view (Alt+Left)"; onClicked: backAction.trigger() }
                    AppButton { objectName: "app_plot_forward"; visible: window.plotWorkflow; colors: window.colors; text: "›"; compact: true; description: "Next plot view (Alt+Right)"; onClicked: forwardAction.trigger() }
                    AppButton { objectName: "app_plot_pan"; visible: window.plotWorkflow; colors: window.colors; text: "Pan"; compact: true; description: "Pan the plot"; checkable: true; checked: window.state.plotTool === "pan"; onClicked: panAction.trigger() }
                    AppButton { objectName: "app_plot_zoom"; visible: window.plotWorkflow; colors: window.colors; text: "Zoom"; compact: true; description: "Zoom into a plot region"; checkable: true; checked: window.state.plotTool === "zoom"; onClicked: zoomAction.trigger() }
                    AppButton { objectName: "app_peak_analysis"; colors: window.colors; text: peakAnalysisAction.text; compact: true; visible: !window.compact && ["pl", "drr", "cmp", "power"].indexOf(window.currentWorkflow) >= 0; enabled: peakAnalysisAction.enabled; description: "Open current plotted data in the independent Peak Analysis window"; onClicked: peakAnalysisAction.trigger() }
                    AppButton { objectName: "drr_plot_raw"; colors: window.colors; text: "ΔR/R"; compact: true; visible: window.currentWorkflow === "drr" && !window.compact; checkable: true; checked: window.state.drrView === "raw"; onClicked: window.appBridge.setParameter("drrView", "raw") }
                    AppButton { objectName: "drr_plot_second"; colors: window.colors; text: "Derivative"; compact: true; visible: window.currentWorkflow === "drr" && !window.compact; enabled: window.state.derivativeAvailable === true; checkable: true; checked: window.state.drrView === "second"; onClicked: window.appBridge.setParameter("drrView", "second") }
                    AppButton { objectName: "drr_plot_pair"; colors: window.colors; text: "Side by side"; compact: true; visible: window.currentWorkflow === "drr" && !window.compact; enabled: window.state.derivativeAvailable === true; checkable: true; checked: window.state.drrView === "pair"; onClicked: window.appBridge.setParameter("drrView", "pair") }
                    AppComboBox {
                        colors: window.colors
                        objectName: "drr_plot_view_compact"
                        visible: window.currentWorkflow === "drr" && window.compact
                        Layout.preferredWidth: 110
                        implicitHeight: 28
                        model: ["ΔR/R", "Derivative", "Side by side"]
                        currentIndex: window.state.drrView === "pair" ? 2 : window.state.drrView === "second" ? 1 : 0
                        onActivated: window.appBridge.setParameter("drrView", ["raw", "second", "pair"][currentIndex])
                    }
                    Item { Layout.fillWidth: true; Layout.minimumWidth: 0 }
                    AppButton {
                        objectName: "app_workflow_selector"
                        colors: window.colors
                        text: window.workflowLabel() + " ▾"
                        visible: !window.railVisible
                        description: "Switch workflow. Current: " + window.workflowLabel()
                        onClicked: compactWorkflowMenu.open()
                    }
                    Controls.Menu {
                        id: compactWorkflowMenu
                        y: 34
                        Repeater {
                            model: window.workflows
                            delegate: Controls.MenuItem {
                                required property var modelData
                                text: modelData.label
                                checkable: true
                                checked: window.currentWorkflow === modelData.key
                                onTriggered: window.appBridge.selectWorkflow(modelData.key)
                            }
                        }
                    }
                    AppButton { objectName: "app_toggle_rail"; colors: window.colors; text: "Workflows"; description: "Show or hide workflow navigation (Ctrl+Shift+B)"; checkable: true; checked: window.railVisible; onClicked: railAction.trigger() }
                    AppButton { objectName: "app_toggle_controls"; colors: window.colors; text: "Controls"; description: "Show or hide workflow controls (Ctrl+B)"; checkable: true; checked: window.controlsVisible; onClicked: controlsAction.trigger() }
                }
            }
            Basic.SplitView {
                objectName: "app_vertical_splitter"
                Layout.fillWidth: true
                Layout.fillHeight: true
                orientation: Qt.Vertical
                handle: Rectangle {
                    implicitHeight: 5
                    color: Basic.SplitHandle.pressed ? window.colors.accent : Basic.SplitHandle.hovered ? window.colors.border : window.colors.divider
                }
                Basic.SplitView {
                    objectName: "app_workspace_splitter"
                    Basic.SplitView.fillHeight: true
                    orientation: Qt.Horizontal
                    handle: Rectangle {
                        implicitWidth: 5
                        color: Basic.SplitHandle.pressed ? window.colors.accent : Basic.SplitHandle.hovered ? window.colors.border : window.colors.divider
                    }
                    Item {
                        objectName: "app_parameter_pane"
                        visible: window.controlsVisible
                        Basic.SplitView.preferredWidth: 320
                        Basic.SplitView.minimumWidth: 220
                        Basic.SplitView.maximumWidth: Math.max(220, Math.min(520, window.width * 0.6))
                        Loader {
                            id: parameterLoader
                            objectName: "app_parameter_loader"
                            anchors.fill: parent
                            visible: status === Loader.Ready
                        }
                        AppParameterPane {
                            objectName: "app_default_parameter_pane"
                            anchors.fill: parent
                            visible: parameterLoader.status !== Loader.Ready
                            bridge: window.appBridge
                            colors: window.colors
                            groups: String(window.appBridge.parameterUrl).length === 0 || parameterLoader.status === Loader.Error
                                    ? (window.state.parameterGroups || []) : []
                            message: parameterLoader.status === Loader.Error
                                     ? "This workflow’s native controls could not be loaded."
                                     : window.state.parameterMessage || ""
                        }
                    }
                    Rectangle {
                        id: workspace
                        objectName: "app_workspace"
                        Basic.SplitView.fillWidth: true
                        Basic.SplitView.minimumWidth: 160
                        color: window.colors.control
                        Loader {
                            id: workspaceLoader
                            objectName: "app_workspace_loader"
                            anchors.fill: parent
                            visible: status === Loader.Ready && !window.state.workspaceMessage
                        }
                        PlotSurface {
                            id: plotSurface
                            objectName: "app_plot_surface"
                            anchors.fill: parent
                            visible: workspaceLoader.status !== Loader.Ready && workspaceLoader.status !== Loader.Error && !window.state.workspaceMessage
                            backend: workspaceLoader.status === Loader.Ready ? null : window.appBridge.plotBackend
                            Accessible.name: "Scientific plot workspace"
                        }
                        AppInspectionDivider {
                            anchors.fill: plotSurface
                            visible: plotSurface.visible && window.state.inspectionVisible === true
                            bridge: window.appBridge
                            colors: window.colors
                            state: window.state
                        }
                        Text {
                            objectName: "app_workspace_message"
                            anchors.fill: parent
                            anchors.margins: 28
                            visible: !!window.state.workspaceMessage || workspaceLoader.status === Loader.Error
                            text: workspaceLoader.status === Loader.Error ? "The native workspace could not be loaded."
                                                                        : window.state.workspaceMessage || ""
                            color: window.colors.muted
                            wrapMode: Text.WordWrap
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            Accessible.name: text
                        }
                    }
                }
                AppBottomPanel {
                    id: bottomPanel
                    objectName: "app_bottom_panel"
                    visible: window.resultsVisible || window.logVisible
                    Basic.SplitView.preferredHeight: 180
                    Basic.SplitView.minimumHeight: 90
                    Basic.SplitView.maximumHeight: Math.max(90, window.height * 0.55)
                    colors: window.colors
                    resultsAvailable: window.resultsVisible
                    logAvailable: window.logVisible
                    results: window.state.resultText || ""
                    log: window.state.logText || ""
                    onCloseRequested: function(panel) { window.invoke(panel === "results" ? "toggle-results" : "toggle-log") }
                }
            }
        }
    }

    footer: Basic.ToolBar {
        objectName: "app_status_bar"
        padding: 4
        background: Rectangle { color: window.colors.inset; border.color: window.colors.divider }
        RowLayout {
            anchors.fill: parent
            spacing: 8
            AppButton { objectName: "app_toggle_results"; colors: window.colors; text: "Results"; description: "Show or hide analysis results (Ctrl+J)"; checkable: true; checked: window.resultsVisible; onClicked: resultsAction.trigger() }
            AppButton { objectName: "app_toggle_log"; colors: window.colors; text: "Log"; description: "Show or hide run log (Ctrl+Shift+J)"; checkable: true; checked: window.logVisible; onClicked: logAction.trigger() }
            Text {
                objectName: "app_status_text"
                Layout.fillWidth: true
                text: window.state.status || "Ready"
                color: window.colors.muted
                elide: Text.ElideRight
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }
            Basic.ProgressBar {
                objectName: "app_progress"
                Layout.preferredWidth: window.compact ? 90 : 150
                visible: window.state.busy === true
                indeterminate: window.state.progress === undefined || window.state.progress < 0
                value: window.state.progress === undefined ? 0 : window.state.progress
                Accessible.name: "Operation progress"
            }
            AppButton { objectName: "app_cancel_operation"; colors: window.colors; text: "Cancel"; visible: window.state.busy === true && window.state.cancellable === true; description: "Cancel the running operation"; onClicked: window.invoke("cancel") }
            AppButton { objectName: "app_update_available"; colors: window.colors; text: window.state.updateText || "Update available"; visible: window.state.updateAvailable === true; description: "View the available application update"; onClicked: window.invoke("show-update") }
            Text { text: window.workflowLabel(); color: window.colors.muted; visible: window.width >= 800 }
        }
    }
}
