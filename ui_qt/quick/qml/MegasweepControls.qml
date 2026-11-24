pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window

Basic.Control {
    id: panel
    objectName: "quick_megasweep_controls"
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
    Connections {
        target: panel.bridge
        ignoreUnknownSignals: true
        function onOperationCompleted(kind, elapsed) {
            if (kind === "background-preview" && panel.state.reflectionPreviewVisible)
                panel.backgroundOpen = false
        }
    }
    implicitWidth: 320
    background: Rectangle { color: panel.colors.surface }
    Accessible.name: "Megasweep processing parameters"
    Basic.Dialog {
        id: backgroundDialog
        objectName: "megasweep_background_dialog"
        parent: panel.Window.window ? panel.Window.window.contentItem : panel
        anchors.centerIn: parent
        width: Math.min(1000, parent.width - 24)
        height: Math.min(640, parent.height - 24)
        title: "Background references"
        modal: true
        visible: panel.backgroundOpen && panel.state.mode === "Reflection"
        onAboutToShow: referenceTabs.currentIndex = 0
        onClosed: panel.backgroundOpen = false
        background: Rectangle { color: panel.colors.surface; border.color: panel.colors.border; radius: 6 }
        contentItem: ColumnLayout {
            spacing: 7
            Basic.TabBar {
                id: referenceTabs
                objectName: "megasweep_background_tabs"
                Layout.fillWidth: true
                Layout.maximumWidth: 320
                background: Rectangle { color: panel.colors.border }
                AppTabButton { objectName: "megasweep_background_files_tab"; colors: panel.colors; text: "Files" }
                AppTabButton { objectName: "megasweep_background_normalization_tab"; colors: panel.colors; text: "Normalization" }
            }
            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: referenceTabs.currentIndex
                ColumnLayout {
                    spacing: 7
                    RowLayout {
                        Layout.fillWidth: true
                        Basic.Label { text: "Reference role"; color: panel.colors.text }
                        AppComboBox {
                            objectName: "megasweep_background_role"
                            Layout.fillWidth: true
                            colors: panel.colors
                            enabled: !panel.state.busy
                            model: panel.roleOptions.map(option => option.label + " · " + panel.roleCount(option.value))
                            currentIndex: Math.max(0, panel.roleOptions.findIndex(option => option.value === panel.currentRole))
                            Accessible.name: "Reference role and number of selected files"
                            onActivated: index => panel.bridge.setParameter("background_role", panel.roleOptions[index].value)
                        }
                    }
                    Basic.Label { objectName: "megasweep_background_role_help"; Layout.fillWidth: true; text: "The selected role is used for all scan points. Assign unknown filenames to a role manually and verify their spectra."; color: panel.colors.muted; wrapMode: Text.Wrap }
                    Basic.Label { objectName: "megasweep_background_status"; Layout.fillWidth: true; text: panel.state.backgroundSelectionError || panel.state.error || panel.state.status || ""; color: (panel.state.backgroundSelectionError || panel.state.error) ? panel.colors.error : panel.colors.muted; visible: text.length > 0; wrapMode: Text.Wrap }
                    BackgroundFileBrowser { objectName: "megasweep_background_browser"; Layout.fillWidth: true; Layout.fillHeight: true; bridge: panel.bridge ? panel.bridge.backgroundSelectionBridge : null; colors: panel.colors; chosenTitle: panel.roleLabel(panel.currentRole) + " backgrounds" }
                }
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    AppParameterPane {
                        objectName: "megasweep_background_normalization"
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        width: Math.min(480, parent.width)
                        pinDataGroup: false
                        bridge: panel.bridge
                        colors: panel.colors
                        groups: panel.state.referenceParameterGroups || []
                        Accessible.name: "Background normalization"
                    }
                }
            }
            AppButton { objectName: "megasweep_background_close"; Layout.alignment: Qt.AlignRight; colors: panel.colors; text: "Close"; onClicked: backgroundDialog.close() }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Basic.TabBar {
            id: modes
            objectName: "megasweep_modes"
            Layout.fillWidth: true
            Layout.margins: 8
            enabled: !panel.state.busy
            currentIndex: panel.state.mode === "Reflection" ? 1 : panel.state.mode === "Transport" ? 2 : 0
            background: Rectangle { color: panel.colors.border }
            AppTabButton { objectName: "megasweep_mode_pl"; colors: panel.colors; text: "PL"; onClicked: panel.bridge.setParameter("mode", "PL") }
            AppTabButton { objectName: "megasweep_mode_reflection"; colors: panel.colors; text: "Reflection"; onClicked: panel.bridge.setParameter("mode", "Reflection") }
            AppTabButton { objectName: "megasweep_mode_transport"; colors: panel.colors; text: "Transport"; onClicked: panel.bridge.setParameter("mode", "Transport") }
        }
        Flow {
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 8
            Layout.bottomMargin: 6
            spacing: 4
            AppButton {
                objectName: "megasweep_select_files"
                colors: panel.colors
                compact: true
                text: panel.state.mode === "Reflection" ? "Open scan…" : "Open file…"
                enabled: !panel.state.busy
                onClicked: panel.bridge.invoke("select-files")
            }
            AppButton {
                objectName: "megasweep_background"
                colors: panel.colors
                compact: true
                visible: panel.state.mode === "Reflection"
                text: "Background: " + panel.roleLabel(panel.currentRole) + " (" + panel.roleCount(panel.currentRole) + ")"
                checkable: true
                checked: panel.backgroundOpen
                onClicked: {
                    panel.backgroundOpen = !panel.backgroundOpen
                    if (panel.backgroundOpen) panel.bridge.invoke("choose-background")
                }
            }
            AppButton {
                objectName: "megasweep_update"
                colors: panel.colors
                compact: true
                text: "Update"
                enabled: !panel.state.busy && !!panel.state.sourceFile
                onClicked: panel.bridge.invoke("update")
            }
        }
        AppParameterPane {
            objectName: "megasweep_scan_parameters"
            Layout.fillWidth: true
            Layout.fillHeight: true
            pinDataGroup: false
            bridge: panel.bridge
            colors: panel.colors
            groups: (panel.state.parameterGroups || []).map(group => Object.assign({}, group,
                {fields: (group.fields || []).filter(field => field.key !== "mode"
                    && !(panel.state.mode === "Reflection" && field.key === "view"))}))
            message: panel.state.error || ""
        }
    }
}
