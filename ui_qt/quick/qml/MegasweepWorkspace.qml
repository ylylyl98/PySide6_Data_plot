pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window
import Dptk 1.0

Rectangle {
    id: page
    objectName: "quick_megasweep_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    readonly property var roleOptions: state.backgroundRoleOptions || [{value:"custom",label:"Custom"},{value:"p",label:"P"},{value:"n",label:"N"},{value:"bn",label:"BN"}]
    readonly property string currentRole: state.backgroundRole || "custom"
    readonly property var roleGroups: state.backgroundGroups || ({})
    function roleLabel(role) { for (let option of roleOptions) if (option.value === role) return option.label; return role }
    function roleCount(role) { return (roleGroups[role] || []).length }
    property bool backgroundOpen: false
    property string lastMode: state.mode || ""
    onStateChanged: { if ((state.mode || "") !== lastMode) { backgroundOpen = false; lastMode = state.mode || "" } }
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "Megasweep plot and background references"
    onColorsChanged: if (bridge) bridge.setPalette(colors)
    Component.onCompleted: if (bridge) bridge.setPalette(colors)

    MegasweepMovieDialog { id: movieDialog; parent: page.Window.window ? page.Window.window.contentItem : page; bridge: page.bridge; colors: page.colors }
    Basic.Dialog {
        id: backgroundDialog
        objectName: "megasweep_background_dialog"
        parent: page.Window.window ? page.Window.window.contentItem : page
        anchors.centerIn: parent
        width: Math.min(1000, parent.width - 24)
        height: Math.min(640, parent.height - 24)
        title: "Background references"
        modal: true
        visible: page.backgroundOpen && page.state.mode === "Reflection"
        onClosed: page.backgroundOpen = false
        background: Rectangle { color: page.colors.surface; border.color: page.colors.border; radius: 6 }
        contentItem: ColumnLayout {
            spacing: 7
            RowLayout {
                Layout.fillWidth: true
                Basic.Label { text: "Reference role"; color: page.colors.text }
                AppComboBox {
                    id: roleSelector
                    objectName: "megasweep_background_role"
                    Layout.fillWidth: true
                    colors: page.colors
                    enabled: !page.state.busy
                    model: page.roleOptions.map(option => option.label + " · " + page.roleCount(option.value))
                    currentIndex: Math.max(0, page.roleOptions.findIndex(option => option.value === page.currentRole))
                    Accessible.name: "Reference role and number of selected files"
                    onActivated: index => page.bridge.setParameter("background_role", page.roleOptions[index].value)
                }
            }
            Basic.Label { objectName: "megasweep_background_role_help"; Layout.fillWidth: true; text: "The selected role is used for all scan points. Assign unknown filenames to a role manually and verify their spectra."; color: page.colors.muted; wrapMode: Text.Wrap }
            Basic.Label { objectName: "megasweep_background_status"; Layout.fillWidth: true; text: page.state.backgroundSelectionError || page.state.error || page.state.status || ""; color: (page.state.backgroundSelectionError || page.state.error) ? page.colors.error : page.colors.muted; visible: text.length > 0; wrapMode: Text.Wrap }
            BackgroundFileBrowser { objectName: "megasweep_background_browser"; Layout.fillWidth: true; Layout.fillHeight: true; bridge: page.bridge ? page.bridge.backgroundSelectionBridge : null; colors: page.colors; chosenTitle: page.roleLabel(page.currentRole) + " backgrounds" }
            AppButton { objectName: "megasweep_background_close"; Layout.alignment: Qt.AlignRight; colors: page.colors; text: "Close"; onClicked: backgroundDialog.close() }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Flow {
            Layout.fillWidth: true
            Layout.margins: 7
            spacing: 4
            AppButton { objectName: "megasweep_select_files"; colors: page.colors; text: "Open file…"; enabled: !page.state.busy; onClicked: page.bridge.invoke("select-files") }
            AppButton { objectName: "megasweep_update"; colors: page.colors; text: "Update"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("update") }
            AppButton { objectName: "megasweep_background"; colors: page.colors; text: "Background: " + page.roleLabel(page.currentRole) + " (" + page.roleCount(page.currentRole) + ")"; visible: page.state.mode === "Reflection"; checkable: true; checked: page.backgroundOpen; onClicked: { page.backgroundOpen = !page.backgroundOpen; if (page.backgroundOpen) page.bridge.invoke("choose-background") } }
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
