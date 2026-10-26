pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Popup {
    id: dialog
    required property var updateBridge
    required property var colors
    readonly property var state: updateBridge.snapshot
    property bool showNotes: false
    objectName: "quick_update_dialog"
    width: Math.min(620, parent ? parent.width - 48 : 620)
    height: Math.min(showNotes ? 550 : state.stage === "about" ? 220 : 440, parent ? parent.height - 48 : 550)
    x: parent ? (parent.width - width) / 2 : 0
    y: parent ? (parent.height - height) / 2 : 0
    visible: state.dialogOpen === true
    modal: true; focus: true
    padding: 20
    closePolicy: Basic.Popup.CloseOnEscape
    onClosed: { showNotes = false; updateBridge.invoke("close-dialog") }
    background: Rectangle { color: dialog.colors.surface; border.color: dialog.colors.border; border.width: 1; radius: 8 }
    contentItem: ColumnLayout {
        spacing: 12
        Basic.Label { Layout.fillWidth: true; text: dialog.state.title || "Updates"; font.pixelSize: 19; font.weight: Font.DemiBold; color: dialog.colors.text; wrapMode: Text.WordWrap }
        Basic.Label { Layout.fillWidth: true; text: dialog.state.message || ""; textFormat: Text.PlainText; color: dialog.state.error ? dialog.colors.error : dialog.colors.text; wrapMode: Text.WordWrap }
        Basic.Label {
            Layout.fillWidth: true
            visible: dialog.state.stage !== "about"
            text: "Installed: " + dialog.state.currentVersion + (dialog.state.latestVersion ? " · Latest stable: " + dialog.state.latestVersion : "")
            color: dialog.colors.muted; wrapMode: Text.WordWrap
        }
        Basic.Label {
            Layout.fillWidth: true
            visible: !!dialog.state.installerPath
            text: dialog.state.installerPath || ""; textFormat: Text.PlainText
            color: dialog.colors.muted; wrapMode: Text.WrapAnywhere
        }
        Basic.ProgressBar {
            Layout.fillWidth: true
            visible: dialog.state.busy
            indeterminate: dialog.state.stage !== "downloading" || !dialog.state.progress
            from: 0; to: 100; value: dialog.state.progress || 0
            Accessible.name: dialog.state.message || "Update progress"
        }
        Basic.CheckBox {
            objectName: "quick_updates_automatic"
            text: "Check automatically for updates"
            checked: dialog.state.automaticChecksEnabled === true
            visible: dialog.state.stage !== "about" && dialog.state.stage !== "confirm-install" && dialog.state.stage !== "launch-verifying"
            onToggled: dialog.updateBridge.setAutomatic(checked)
            contentItem: Text { text: parent.text; font: parent.font; color: dialog.colors.text; leftPadding: parent.indicator.width + parent.spacing; verticalAlignment: Text.AlignVCenter; wrapMode: Text.WordWrap }
        }
        Basic.ScrollView {
            Layout.fillWidth: true; Layout.fillHeight: true
            visible: dialog.showNotes && !!dialog.state.releaseNotes
            contentWidth: availableWidth
            Basic.TextArea {
                text: dialog.state.releaseNotes || ""
                textFormat: TextEdit.PlainText
                readOnly: true; selectByMouse: true; wrapMode: TextEdit.Wrap
                color: dialog.colors.text
                background: Rectangle { color: dialog.colors.control; radius: 4 }
            }
        }
        Item { Layout.fillHeight: true; visible: !dialog.showNotes || !dialog.state.releaseNotes }
        Flow {
            Layout.fillWidth: true; spacing: 7
            AppButton {
                objectName: "quick_update_notes"
                colors: dialog.colors; text: "Release notes"
                visible: !!dialog.state.releaseNotes && dialog.state.stage !== "about"
                checked: dialog.showNotes
                onClicked: dialog.showNotes = !dialog.showNotes
            }
            AppButton {
                objectName: "quick_update_release_page"
                colors: dialog.colors; text: "Open release page"
                visible: !!dialog.state.releaseUrl && dialog.state.stage !== "about"
                enabled: !dialog.state.busy
                onClicked: dialog.updateBridge.invoke("release-notes")
            }
            AppButton {
                objectName: "quick_update_download"
                colors: dialog.colors; text: "Download update"; primary: true
                visible: dialog.state.canDownload === true
                onClicked: dialog.updateBridge.invoke("download")
            }
            AppButton {
                objectName: "quick_update_install"
                colors: dialog.colors; text: "Install now…"; primary: true
                visible: dialog.state.canInstall === true
                onClicked: dialog.updateBridge.invoke("install")
            }
            AppButton {
                objectName: "quick_update_confirm_install"
                colors: dialog.colors; text: "Launch installer"; primary: true
                visible: dialog.state.installConfirmationRequired === true
                onClicked: dialog.updateBridge.invoke("confirm-install")
            }
            AppButton {
                objectName: "quick_update_open_folder"
                colors: dialog.colors; text: "Open folder"
                visible: dialog.state.canOpenFolder === true
                enabled: !dialog.state.busy
                onClicked: dialog.updateBridge.invoke("open-folder")
            }
            AppButton {
                objectName: "quick_update_later"
                colors: dialog.colors
                text: dialog.state.stage === "about" || dialog.state.stage === "error" || dialog.state.stage === "up_to_date" ? "Close" : "Later"
                onClicked: dialog.updateBridge.invoke("later")
            }
        }
    }
}
