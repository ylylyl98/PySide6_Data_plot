pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Dialog {
    id: panel
    objectName: "drr_analysis_library"
    required property var bridge
    required property var colors
    title: "Processed files / comparison members / saved P2P ranges"
    modal: true
    standardButtons: Basic.Dialog.Close
    width: Math.min(980, parent ? parent.width - 40 : 980)
    height: Math.min(700, parent ? parent.height - 40 : 700)
    anchors.centerIn: parent
    background: Rectangle { color: panel.colors.surface; border.color: panel.colors.border; radius: 8 }
    contentItem: ColumnLayout {
        spacing: 8
        Basic.TabBar {
            id: tabs
            objectName: "drr_analysis_library_tabs"
            Layout.fillWidth: true
            AppTabButton { colors: panel.colors; text: "Processed files" }
            AppTabButton { colors: panel.colors; text: "Comparison members" }
            AppTabButton { colors: panel.colors; text: "Saved P2P ranges" }
        }
        Basic.Label {
            Layout.fillWidth: true
            text: tabs.currentIndex === 0 ? "Select independent saved DAT products. Source files are preserved."
                : tabs.currentIndex === 1 ? "Check members to include. Different exposures / averaging / backgrounds may affect comparison."
                : "Restore a previous common calculation range and its cached results."
            color: panel.colors.muted
            wrapMode: Text.Wrap
        }
        Basic.TextField {
            Layout.fillWidth: true
            visible: tabs.currentIndex === 0
            placeholderText: "Filter processed files…"
            text: panel.bridge.snapshot.catalogQuery
            onTextEdited: panel.bridge.setParameter("catalogQuery", text)
        }
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabs.currentIndex
            ListView {
                id: processed
                clip: true
                model: panel.bridge.catalogModel
                Basic.ScrollBar.vertical: Basic.ScrollBar {}
                delegate: Basic.CheckBox {
                    required property var entry
                    width: processed.width
                    text: entry.name
                    checked: entry.selected
                    Accessible.name: "Include processed file " + entry.name
                    onToggled: panel.bridge.setParameter("processed:" + entry.id, checked)
                    Basic.ToolTip.visible: hovered
                    Basic.ToolTip.text: entry.path
                }
            }
            ListView {
                id: members
                clip: true
                model: panel.bridge.membersModel
                Basic.ScrollBar.vertical: Basic.ScrollBar {}
                delegate: Basic.CheckBox {
                    required property var entry
                    width: members.width
                    text: entry.field + " T · " + entry.name + "\n" + entry.processing
                    checked: entry.included
                    Accessible.name: "Include comparison member " + entry.name
                    onToggled: panel.bridge.setMember(entry.id, checked)
                }
            }
            ListView {
                id: history
                clip: true
                model: panel.bridge.historyModel
                Basic.ScrollBar.vertical: Basic.ScrollBar {}
                delegate: AppItemDelegate {
                    colors: panel.colors
                    required property var entry
                    required property int index
                    width: history.width
                    text: entry.name
                    highlighted: ListView.isCurrentItem
                    onClicked: { history.currentIndex = index; panel.bridge.setParameter("selectedHistory", entry.id) }
                    onDoubleClicked: { panel.bridge.restoreWorkspace(entry.id); panel.close() }
                }
            }
        }
        Flow {
            Layout.fillWidth: true
            spacing: 6
            AppButton { colors: panel.colors; visible: tabs.currentIndex === 0; text: "Add selected products"; onClicked: panel.bridge.invoke("load_processed") }
            AppButton { colors: panel.colors; visible: tabs.currentIndex === 0; text: "Browse products…"; onClicked: panel.bridge.invoke("add_processed") }
            AppButton { colors: panel.colors; visible: tabs.currentIndex === 1; text: "Add other files…"; onClicked: panel.bridge.invoke("add_members") }
            AppButton { colors: panel.colors; visible: tabs.currentIndex === 1; text: "Restore automatic"; onClicked: panel.bridge.invoke("reset_members") }
            AppButton { colors: panel.colors; visible: tabs.currentIndex === 1; text: "Apply / open group"; onClicked: { panel.bridge.invoke("apply_members"); panel.close() } }
            AppButton { colors: panel.colors; visible: tabs.currentIndex === 2; text: "Restore selected range"; onClicked: { panel.bridge.invoke("restore_history"); panel.close() } }
        }
    }
}
