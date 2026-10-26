pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import "DialogState.js" as DialogState

Basic.ApplicationWindow {
    id: window
    objectName: "quick_organizer_window"
    required property var organizerBridge
    readonly property var state: organizerBridge.snapshot
    readonly property var colors: organizerBridge.colors
    property string folderRole: "experiment-folder"
    title: "MCD Organizer"
    width: 1380
    height: 900
    minimumWidth: 760
    minimumHeight: 600
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
    onClosing: function(event) { if (!state.canClose) { event.accepted = false } else organizerBridge.close() }
    header: Basic.ToolBar {
        background: Rectangle { color: window.colors.surface }
        RowLayout {
            anchors.fill: parent
            AppButton { colors: window.colors; text: "Experiment…"; onClicked: window.organizerBridge.invoke("choose-experiment") }
            AppButton { colors: window.colors; text: "Refresh"; onClicked: window.organizerBridge.invoke("refresh") }
            AppButton { colors: window.colors; text: "Export checked series"; primary: true; enabled: window.state.canExport; onClicked: window.organizerBridge.invoke("export") }
            Text { Layout.fillWidth: true; text: window.state.summary || ""; color: window.colors.muted; elide: Text.ElideRight }
            AppButton { colors: window.colors; text: "Light"; onClicked: window.organizerBridge.invoke("appearance-light") }
            AppButton { colors: window.colors; text: "Dark"; onClicked: window.organizerBridge.invoke("appearance-dark") }
        }
    }
    Basic.SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal
        handle: Rectangle { implicitWidth: 5; color: window.colors.divider }
        ColumnLayout {
            Basic.SplitView.preferredWidth: 320
            Basic.SplitView.minimumWidth: 220
            spacing: 2
            AppParameterPane { Layout.fillWidth: true; Layout.fillHeight: true; Layout.preferredHeight: 420; bridge: window.organizerBridge; colors: window.colors; groups: window.state.parameterGroups || [] }
            Text { Layout.fillWidth: true; Layout.margins: 6; text: "Processed condition series"; color: window.colors.text; font.weight: Font.DemiBold }
            Flow { Layout.fillWidth: true; spacing: 4; AppButton { colors: window.colors; compact: true; text: "Check all"; onClicked: window.organizerBridge.invoke("check-all") } AppButton { colors: window.colors; compact: true; text: "Clear checks"; onClicked: window.organizerBridge.invoke("clear-checked") } }
            Basic.ScrollView {
                id: seriesScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredHeight: 260
                contentWidth: availableWidth
                clip: true
                ColumnLayout {
                    width: seriesScroll.availableWidth
                    Repeater {
                        model: window.state.series || []
                        delegate: RowLayout {
                            id: seriesRow
                            required property var modelData
                            Layout.fillWidth: true
                            Basic.CheckBox { checked: seriesRow.modelData.checked; onToggled: window.organizerBridge.includeSeries(seriesRow.modelData.id, checked); Accessible.name: "Export " + seriesRow.modelData.label }
                            AppButton { Layout.fillWidth: true; colors: window.colors; text: seriesRow.modelData.label; description: seriesRow.modelData.warning || seriesRow.modelData.label; checked: window.state.selectedSeries === seriesRow.modelData.id; onClicked: window.organizerBridge.selectSeries(seriesRow.modelData.id) }
                        }
                    }
                }
            }
        }
        OrganizerWorkspace { Basic.SplitView.fillWidth: true; Basic.SplitView.minimumWidth: 420; bridge: window.organizerBridge; colors: window.colors }
    }
    footer: Text { padding: 5; text: window.state.saveError || window.state.error || window.state.status || ""; color: window.state.error ? window.colors.error : window.colors.muted; elide: Text.ElideRight }
    AppFolderDialog {
        colors: window.colors
        parentWindow: window
        id: folderDialog
        title: window.folderRole === "experiment-folder" ? "Choose experiment or processed MCD folder" : "Choose Organizer results folder"
        options: FolderDialog.DontUseNativeDialog
        onAccepted: window.organizerBridge.openFiles(window.folderRole, [selectedFolder])
    }
    Connections {
        target: window.organizerBridge
        function onSourceDialogRequested(kind, multiple) {
            window.folderRole = kind
            DialogState.prepareFolder(folderDialog, kind === "experiment-folder" ? window.state.folderUrl : window.state.outputUrl)
            folderDialog.open()
        }
    }
}
