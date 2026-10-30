pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Popup {
    id: dialog
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.viewState : ({})
    // Selection/status snapshots must not reset the list model or its scroll.
    readonly property string inventorySignature: visible ? JSON.stringify(state.catalog || []) : ""
    readonly property var catalogRows: inventorySignature ? JSON.parse(inventorySignature) : []
    readonly property string statusFilter: state.values?.historyFilter || "all"
    readonly property var files: catalogRows.filter(function(row) {
        const terms = search.text.toLowerCase().trim().split(/\s+/).filter(function(term) { return !!term })
        const name = (row.name + " " + row.path).toLowerCase()
        return (dialog.statusFilter === "all" || row.status === dialog.statusFilter) && terms.every(function(term) { return name.indexOf(term) >= 0 })
    })
    objectName: "mcd_source_catalog"
    parent: Basic.Overlay.overlay
    width: Math.max(280, Math.min(1100, parent ? parent.width - 32 : 900))
    height: Math.max(280, Math.min(760, parent ? parent.height - 32 : 650))
    anchors.centerIn: parent
    visible: state.sheet === "sources"
    modal: true
    focus: true
    padding: 12
    closePolicy: Basic.Popup.CloseOnEscape
    onOpened: { search.text = ""; search.forceActiveFocus() }
    onClosed: if (bridge && state.sheet === "sources") bridge.invoke("close-sheet")
    background: Rectangle { color: dialog.colors.surface; border.color: dialog.colors.border; radius: 8 }

    ColumnLayout {
        anchors.fill: parent
        spacing: 8
        RowLayout {
            Layout.fillWidth: true
            Text { Layout.fillWidth: true; text: "Choose MCD measurement"; color: dialog.colors.text; font.pointSize: 12; font.weight: Font.DemiBold }
            AppButton { colors: dialog.colors; compact: true; text: "Close"; onClicked: dialog.bridge.invoke("close-sheet") }
        }
        Text { Layout.fillWidth: true; text: dialog.state.sourcePath || "Browse to choose an MCD sweep CSV."; color: dialog.colors.muted; wrapMode: Text.WrapAnywhere }
        RowLayout {
            Layout.fillWidth: true
            Basic.TextField {
                id: search
                objectName: "mcd_source_search"
                Layout.fillWidth: true
                selectByMouse: true
                placeholderText: "Search filename or folder…"
                Accessible.name: "Search MCD files"
            }
            AppComboBox {
                objectName: "mcd_source_status"
                colors: dialog.colors
                model: ["All", "New", "Processed", "History unknown"]
                readonly property var values: ["all", "new", "processed", "unknown"]
                currentIndex: values.indexOf(dialog.state.values?.historyFilter || "all")
                Accessible.name: "MCD source status"
                onActivated: dialog.bridge.setParameter("historyFilter", values[currentIndex])
            }
            AppButton { colors: dialog.colors; compact: true; text: "Refresh"; onClicked: dialog.bridge.invoke("refresh-catalog") }
        }
        ListView {
            id: list
            objectName: "mcd_source_files"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 5
            model: dialog.files
            Basic.ScrollBar.vertical: Basic.ScrollBar {}
            delegate: CatalogFileRow {
                id: row
                required property var modelData
                width: list.width
                colors: dialog.colors
                objectName: "mcd_source_file_" + modelData.path
                entry: ({id: modelData.path, name: modelData.name, source: modelData.path,
                         selected: dialog.state.sourceDraft === modelData.path,
                         detail: "Select this MCD sweep, then choose Open selected."})
                onSelectionRequested: selected => dialog.bridge.setSourceDraft(selected ? modelData.path : "")
                Text {
                    objectName: "mcd_source_filename_" + row.modelData.path
                    Layout.fillWidth: true
                    text: row.modelData.name
                    font: row.font
                    color: dialog.colors.text
                    wrapMode: Text.WrapAnywhere
                }
                RowLayout {
                    Layout.fillWidth: true
                    SourceStatusBadge { colors: dialog.colors; status: row.modelData.status }
                    Text { Layout.fillWidth: true; text: row.modelData.processedAt || ""; color: dialog.colors.muted; font.pixelSize: 11 }
                }
            }
            Text {
                anchors.centerIn: parent
                width: parent.width - 24
                visible: list.count === 0 && !dialog.state.catalogBusy
                text: "No matching MCD files. Change the search or status filter, or Browse files…"
                color: dialog.colors.muted
                wrapMode: Text.Wrap
                horizontalAlignment: Text.AlignHCenter
            }
        }
        Text { text: "Selected file · full path (select to copy)"; color: dialog.colors.muted }
        Basic.ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(110, selectedPath.implicitHeight + 4)
            contentWidth: availableWidth
            clip: true
            Basic.TextArea {
                id: selectedPath
                objectName: "mcd_source_full_path"
                readOnly: true
                selectByMouse: true
                text: dialog.state.sourceDraft || "No file selected"
                wrapMode: Text.WrapAnywhere
                color: dialog.colors.text
                background: Rectangle { color: dialog.colors.inset; radius: 4 }
            }
        }
        Basic.ProgressBar { Layout.fillWidth: true; indeterminate: true; visible: dialog.state.catalogBusy }
        Text { Layout.fillWidth: true; visible: !!dialog.state.error; text: dialog.state.error || ""; color: dialog.colors.error; wrapMode: Text.Wrap }
        RowLayout {
            Layout.fillWidth: true
            AppButton { objectName: "mcd_source_browse"; colors: dialog.colors; text: "Browse files…"; onClicked: dialog.bridge.invoke("browse-sources") }
            Item { Layout.fillWidth: true }
            AppButton { objectName: "mcd_source_cancel"; colors: dialog.colors; text: "Cancel"; onClicked: dialog.bridge.invoke("close-sheet") }
            AppButton {
                objectName: "mcd_source_accept"
                colors: dialog.colors
                text: "Open selected"
                primary: true
                enabled: !!dialog.state.sourceDraft && !dialog.state.busy
                onClicked: dialog.bridge.invoke("accept-source")
            }
        }
    }
}
