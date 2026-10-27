pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: browser
    required property var bridge
    required property var colors
    property string chosenTitle: "Selected references"
    readonly property var state: bridge ? bridge.snapshot : ({selected: []})
    objectName: "drr_background_inline_browser"
    spacing: 7
    enabled: state.editEnabled !== false

    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        Basic.TextField {
            objectName: "drr_background_inline_search"
            Layout.fillWidth: true
            implicitHeight: 30
            placeholderText: "Search reference filename / folder…"
            text: browser.state.query || ""
            color: browser.colors.text
            placeholderTextColor: browser.colors.muted
            selectionColor: browser.colors.accent
            selectedTextColor: browser.colors.selectedText
            onTextEdited: browser.bridge.setParameter("query", text)
            background: Rectangle { color: browser.colors.control; radius: 4; border.color: parent.activeFocus ? browser.colors.focus : browser.colors.border }
        }
        Basic.CheckBox {
            objectName: "drr_background_inline_show_all"
            text: "Show all CSV files"
            implicitHeight: 30
            checked: browser.state.showAllFiles === true
            onToggled: browser.bridge.setParameter("showAllFiles", checked)
            contentItem: Text { text: parent.text; color: browser.colors.text; font: parent.font; leftPadding: parent.indicator.width + parent.spacing; verticalAlignment: Text.AlignVCenter }
        }
        AppButton { objectName: "drr_background_inline_refresh"; colors: browser.colors; text: "Refresh"; compact: true; enabled: !browser.state.busy; onClicked: browser.bridge.invoke("refresh") }
        AppButton { objectName: "drr_background_inline_browse"; colors: browser.colors; text: "Browse files…"; compact: true; onClicked: browser.bridge.invoke("browse") }
    }
    Basic.SplitView {
        Layout.fillWidth: true
        Layout.fillHeight: true
        Layout.minimumHeight: 120
        orientation: Qt.Horizontal
        ColumnLayout {
            Basic.SplitView.fillWidth: true
            Basic.SplitView.minimumWidth: 220
            spacing: 5
            ColumnLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: 30
                spacing: 1
                Basic.Label { text: "Available files · " + available.count; color: browser.colors.text; font.weight: Font.DemiBold; font.pixelSize: 12 }
                Basic.Label {
                    Layout.fillWidth: true
                    text: browser.state.folder || ""
                    color: browser.colors.muted
                    elide: Text.ElideMiddle
                    font.pixelSize: 10
                    HoverHandler { id: folderHover }
                    Basic.ToolTip.visible: folderHover.hovered
                    Basic.ToolTip.text: text
                }
            }
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: browser.colors.control
                border.color: browser.colors.border
                radius: 5
                CatalogListView {
                    id: available
                    objectName: "drr_background_available_files"
                    anchors.fill: parent
                    anchors.margins: 4
                    model: browser.bridge ? browser.bridge.filesModel : null
                    clip: true
                    spacing: 5
                    delegate: CatalogFileRow {
                        id: fileRow
                        colors: browser.colors
                        objectName: "drr_background_available_" + entry.name
                        width: available.width
                        onSelectionRequested: selected => browser.bridge.selectFile(entry.id, selected)
                        Text { Layout.fillWidth: true; text: fileRow.entry.name; color: browser.colors.text; font.pixelSize: 12; wrapMode: Text.Wrap; maximumLineCount: 3; elide: Text.ElideRight }
                        Text { Layout.fillWidth: true; text: fileRow.entry.summary || fileRow.entry.source; color: browser.colors.muted; font.pixelSize: 11; wrapMode: Text.Wrap; maximumLineCount: 2; elide: Text.ElideRight }
                    }
                    Basic.Label {
                        anchors.centerIn: parent
                        width: Math.max(120, parent.width - 32)
                        visible: available.count === 0
                        text: browser.state.busy ? "Listing reference files…" : "No matching files.\nTry another search, Show all CSV files, or Browse files."
                        color: browser.colors.muted
                        wrapMode: Text.Wrap
                        horizontalAlignment: Text.AlignHCenter
                    }
                }
            }
        }
        ColumnLayout {
            Basic.SplitView.preferredWidth: browser.width * .4
            Basic.SplitView.minimumWidth: 200
            spacing: 5
            RowLayout {
                Layout.fillWidth: true
                Basic.Label { Layout.fillWidth: true; text: browser.chosenTitle + " · " + chosen.count; color: browser.colors.text; font.weight: Font.DemiBold; elide: Text.ElideRight }
                AppButton { objectName: "drr_background_inline_clear"; colors: browser.colors; text: "Clear all"; compact: true; description: "Remove all references from this draft"; enabled: browser.state.selected.length > 0; onClicked: browser.bridge.invoke("clear") }
            }
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: browser.colors.control
                border.color: browser.colors.border
                radius: 5
                CatalogListView {
                    id: chosen
                    objectName: "drr_background_chosen_files"
                    anchors.fill: parent
                    anchors.margins: 4
                    model: browser.bridge ? browser.bridge.chosenModel : null
                    clip: true
                    spacing: 5
                    delegate: AppItemDelegate {
                        id: chosenRow
                        required property var entry
                        objectName: "drr_background_chosen_" + entry.name
                        colors: browser.colors
                        width: chosen.width
                        focusPolicy: Qt.StrongFocus
                        text: entry.name
                        highlighted: entry.id === browser.state.focused
                        onClicked: browser.bridge.setParameter("focused", entry.id)
                        onDoubleClicked: browser.bridge.selectFile(entry.id, false)
                        Keys.onPressed: event => {
                            if (event.key === Qt.Key_Delete || event.key === Qt.Key_Backspace) {
                                browser.bridge.selectFile(entry.id, false)
                                event.accepted = true
                            }
                        }
                        Basic.ToolTip.visible: hovered
                        Basic.ToolTip.text: entry.source + "\nDouble-click or press Delete to remove from selection."
                        Basic.ToolTip.delay: 650
                        contentItem: RowLayout {
                            spacing: 5
                            ColumnLayout {
                                Layout.fillWidth: true
                                Layout.minimumWidth: 0
                                spacing: 4
                                Text { Layout.fillWidth: true; text: chosenRow.entry.name; color: browser.colors.text; font.pixelSize: 12; wrapMode: Text.Wrap; maximumLineCount: 3; elide: Text.ElideMiddle }
                                Text { Layout.fillWidth: true; text: chosenRow.entry.source; color: browser.colors.muted; font.pixelSize: 11; elide: Text.ElideMiddle }
                            }
                            AppButton { objectName: "drr_background_inline_remove_" + chosenRow.entry.name; colors: browser.colors; text: "Remove"; compact: true; description: "Remove " + chosenRow.entry.name + " from this draft"; onClicked: browser.bridge.selectFile(chosenRow.entry.id, false) }
                        }
                    }
                    Basic.Label { anchors.centerIn: parent; width: parent.width - 24; visible: chosen.count === 0; text: "Click filenames or check files on the left\nto add them to this draft."; color: browser.colors.muted; horizontalAlignment: Text.AlignHCenter; wrapMode: Text.Wrap }
                }
            }
        }
    }
    Basic.ProgressBar { Layout.fillWidth: true; visible: browser.state.busy; indeterminate: true }
    Basic.Label { Layout.fillWidth: true; visible: !!browser.state.error; text: browser.state.error || ""; color: browser.colors.error; wrapMode: Text.Wrap }
}
