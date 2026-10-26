pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import "DialogState.js" as DialogState

Basic.ApplicationWindow {
    id: window
    property var bridge: drrAnalysisBridge
    readonly property var state: bridge.snapshot
    readonly property var colors: bridge.palette
    property bool allowClose: false
    property string dialogKind: ""
    objectName: "drr_analysis_application"
    title: "DRR Analysis"
    width: 1500
    height: 940
    minimumWidth: 900
    minimumHeight: 600
    visible: true
    color: colors.inset
    palette.window: colors.inset
    palette.windowText: colors.text
    palette.base: colors.control
    palette.alternateBase: colors.surface
    palette.text: colors.text
    palette.button: colors.control
    palette.buttonText: colors.text
    palette.brightText: colors.text
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.mid: colors.border
    palette.midlight: colors.pressed
    palette.dark: colors.border
    palette.light: colors.control
    palette.placeholderText: colors.muted
    onClosing: function(event) {
        if (!allowClose) { event.accepted = false; bridge.requestClose() }
    }
    Connections {
        target: window.bridge
        function onCloseReady() { window.allowClose = true; window.close() }
        function onSourceDialogRequested(kind, multiple) {
            window.dialogKind = kind
            if (kind === "folder" || kind === "export_peaks" || kind === "export_summary") {
                DialogState.prepareFolder(folderDialog, kind === "folder" ? window.state.sourceFolderUrl : window.state.exportFolderUrl)
                folderDialog.open()
            }
            else {
                let folder = kind === "save" || kind === "restore" || kind === "snapshot" || kind === "history"
                    ? window.state.sessionFolderUrl : kind.startsWith("export_") ? window.state.exportFolderUrl : window.state.sourceFolderUrl
                DialogState.prepareFile(fileDialog, folder, "")
                fileDialog.fileMode = kind === "save" || kind.startsWith("export_") ? FileDialog.SaveFile : multiple ? FileDialog.OpenFiles : FileDialog.OpenFile
                fileDialog.nameFilters = kind === "save" || kind === "restore" || kind === "snapshot" || kind === "history" ? ["DRR workspace (*.npz)"]
                    : kind === "processed" || kind === "members" ? ["Processed DRR metadata (*.metadata.json)"]
                    : kind.startsWith("export_") ? [kind.substring(7).toUpperCase() + " (*." + kind.substring(7) + ")"] : ["Measurements (*.csv *.dat *.txt)", "All files (*)"]
                fileDialog.defaultSuffix = kind === "save" ? "npz" : kind.startsWith("export_") ? kind.substring(7) : ""
                fileDialog.open()
            }
        }
    }
    AppFileDialog {
        colors: window.colors
        parentWindow: window
        id: fileDialog
        objectName: "drr_analysis_file_dialog"
        options: FileDialog.DontUseNativeDialog
        title: "DRR Analysis"
        onAccepted: window.bridge.openFiles(window.dialogKind, selectedFiles)
        onRejected: window.bridge.dialogRejected(window.dialogKind)
    }
    AppFolderDialog {
        colors: window.colors
        parentWindow: window
        id: folderDialog
        objectName: "drr_analysis_folder_dialog"
        options: FolderDialog.DontUseNativeDialog
        title: "Choose folder"
        onAccepted: window.bridge.openFiles(window.dialogKind, [selectedFolder])
    }
    header: Basic.ToolBar {
        background: Rectangle { color: window.colors.surface }
        Flow {
            width: parent.width
            padding: 7
            spacing: 6
            AppButton { colors: window.colors; text: "Add files…"; onClicked: window.bridge.invoke("add_files") }
            AppButton { colors: window.colors; text: "Add snapshot…"; onClicked: window.bridge.invoke("add_snapshot") }
            AppButton { colors: window.colors; text: "Processed / members / history…"; onClicked: library.open() }
            AppButton { colors: window.colors; text: "Open workspace…"; onClicked: window.bridge.invoke("open_workspace") }
            AppButton { colors: window.colors; text: "Save workspace…"; onClicked: window.bridge.invoke("save_workspace") }
            AppButton { colors: window.colors; text: "Cancel task"; enabled: window.state.busy; onClicked: window.bridge.invoke("cancel") }
        }
    }
    Basic.SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal
        ColumnLayout {
            Basic.SplitView.preferredWidth: 260
            Basic.SplitView.minimumWidth: 170
            Basic.SplitView.maximumWidth: 460
            spacing: 6
            Basic.Label { Layout.fillWidth: true; text: "Magnetic-field comparison groups"; color: window.colors.text; font.bold: true; wrapMode: Text.Wrap }
            RowLayout {
                Layout.fillWidth: true
                AppButton { colors: window.colors; text: "Folder…"; onClicked: window.bridge.invoke("folder") }
                AppButton { colors: window.colors; text: "Refresh"; onClicked: window.bridge.invoke("refresh_catalog") }
            }
            Basic.TextField {
                Layout.fillWidth: true
                placeholderText: "Sample / spot / gate…"
                text: window.state.groupQuery
                onTextEdited: window.bridge.setParameter("groupQuery", text)
            }
            ListView {
                id: groups
                Layout.fillWidth: true
                Layout.preferredHeight: 210
                clip: true
                model: window.bridge.groupsModel
                Basic.ScrollBar.vertical: Basic.ScrollBar {}
                delegate: AppItemDelegate {
                    colors: window.colors
                    required property var entry
                    width: groups.width
                    text: entry.name + "\n" + entry.count + " files · " + entry.fields + " fields"
                    highlighted: window.state.selectedGroup === entry.id
                    onClicked: window.bridge.setParameter("selectedGroup", entry.id)
                    onDoubleClicked: window.bridge.openComparison(entry.id)
                    contentItem: Text { text: parent.text; color: window.colors.text; wrapMode: Text.Wrap; font.pixelSize: 12 }
                }
            }
            AppButton { Layout.fillWidth: true; colors: window.colors; text: "Open selected group"; onClicked: window.bridge.invoke("open_group") }
            Basic.Label { text: "Loaded datasets"; color: window.colors.text; font.bold: true }
            ListView {
                id: datasets
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                model: window.state.datasets
                Basic.ScrollBar.vertical: Basic.ScrollBar {}
                delegate: Rectangle {
                    required property var modelData
                    width: datasets.width
                    height: Math.max(52, datasetLabel.implicitHeight+12)
                    color: modelData.active ? window.colors.selected : "transparent"
                    RowLayout {
                        anchors.fill: parent
                        Basic.CheckBox { checked: modelData.selected; Accessible.name: "Select " + modelData.name; onToggled: window.bridge.selectDataset(modelData.id, checked) }
                        Basic.Label {
                            id: datasetLabel
                            Layout.fillWidth: true
                            text: modelData.name + "\n" + (modelData.analyzed ? "Peaks ✓ " : "Peaks pending ") + (modelData.p2p ? "· P2P ✓" : "")
                            color: window.colors.text
                            wrapMode: Text.Wrap
                            MouseArea { anchors.fill: parent; onClicked: window.bridge.setParameter("activeKey", modelData.id) }
                        }
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true
                AppButton { colors: window.colors; text: "Select all"; onClicked: window.bridge.invoke("select_all") }
                AppButton { colors: window.colors; text: "Remove selected"; onClicked: window.bridge.invoke("remove_selected") }
            }
        }
        DrrAnalysisWorkspace { Basic.SplitView.fillWidth: true; Basic.SplitView.minimumWidth: 380; bridge: window.bridge; colors: window.colors }
        DrrAnalysisParameters { Basic.SplitView.preferredWidth: 320; Basic.SplitView.minimumWidth: 240; Basic.SplitView.maximumWidth: 520; bridge: window.bridge; colors: window.colors }
    }
    footer: ColumnLayout {
        spacing: 2
        Basic.ProgressBar { Layout.fillWidth: true; visible: window.state.busy; from: 0; to: 100; value: window.state.progress; indeterminate: value === 0 }
        Basic.Label {
            Layout.fillWidth: true
            Layout.margins: 5
            text: window.state.error || window.state.status
            color: window.state.error ? window.colors.error : window.colors.muted
            wrapMode: Text.Wrap
            Accessible.name: text
        }
    }
    DrrAnalysisLibrary { id: library; bridge: window.bridge; colors: window.colors; parent: Basic.Overlay.overlay }
}
