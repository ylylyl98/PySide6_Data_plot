pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Dialog {
    id: dialog
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    readonly property bool backgroundSelection: state.backgroundSelection === true
    property bool reopenPositionPending: false
    property string reopenFile: ""
    property string reopenFilters: ""

    function filterSignature() {
        return JSON.stringify([state.query, state.type, state.statusFilter, state.savedOnly,
            state.onlyNew, state.includeBackgrounds, state.includeAll, state.showAllFiles])
    }

    function cancelReopenPosition() {
        reopenPositionPending = false
        reopenTimer.stop()
        bridge.cancelReopenPosition()
    }
    function restoreReopenPosition() {
        if (!visible || !reopenPositionPending)
            return
        // A cold inventory preview may not contain the file or its group yet.
        // Restore once, after the target arrives, without triggering a scan.
        if (state.reopenPositionPending && state.busy)
            return
        if (reopenFile && bridge.filesModel.indexOfId(reopenFile) < 0 && state.busy)
            return
        if (groups.visible)
            groups.revealId(state.selectedGroup)
        files.revealId(reopenFile)
        cancelReopenPosition()
    }
    onOpened: {
        reopenFile = state.focused || (state.selected.length ? state.selected[state.selected.length - 1] : "")
        reopenFilters = filterSignature()
        reopenPositionPending = true
        reopenTimer.restart()
    }
    onClosed: cancelReopenPosition()
    Timer { id: reopenTimer; interval: 0; onTriggered: dialog.restoreReopenPosition() }
    objectName: "quick_source_catalog"
    title: backgroundSelection ? state.selectionTitle : state.savedOnly ? "Saved " + state.workflow.toUpperCase() + " results" : "Choose " + state.workflow.toUpperCase() + " measurements"
    modal: true
    focus: true
    closePolicy: Basic.Popup.CloseOnEscape
    width: Math.min(1150,parent ? parent.width-40 : 1150)
    height: Math.min(790,parent ? parent.height-40 : 790)
    anchors.centerIn: parent
    onRejected: bridge.invoke("cancel")
    background: Rectangle { color: dialog.colors.surface; radius: 8; border.color: dialog.colors.border }
    header: Rectangle {
        implicitHeight: 48
        color: dialog.colors.surface
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 8
            Basic.Label {
                Layout.fillWidth: true
                text: dialog.title
                color: dialog.colors.text
                font.pointSize: 12
                font.weight: Font.DemiBold
                elide: Text.ElideRight
                Accessible.role: Accessible.Heading
            }
            AppButton {
                objectName: "source_catalog_close"
                colors: dialog.colors
                text: "×"
                font.pointSize: 14
                compact: true
                description: dialog.backgroundSelection ? "Close reference selection" : "Close measurement selection"
                onClicked: dialog.reject()
            }
        }
    }
    Connections {
        target: dialog.bridge
        function onSelectionAccepted(workflow,files) { dialog.close() }
        function onSnapshotChanged() {
            if (!dialog.reopenPositionPending)
                return
            if (dialog.state.focused !== dialog.reopenFile || dialog.filterSignature() !== dialog.reopenFilters)
                dialog.cancelReopenPosition()
            else
                reopenTimer.restart()
        }
    }
    contentItem: ColumnLayout {
        spacing: 7
        Basic.Label { Layout.fillWidth: true; text: dialog.state.folder; color: dialog.colors.muted; elide: Text.ElideMiddle }
        RowLayout {
            Layout.fillWidth: true
            Basic.TextField { objectName: "source_catalog_search"; Layout.fillWidth: true; placeholderText: dialog.backgroundSelection ? "Search reference filename / folder…" : "Search filename / gate range / conditions…"; text: dialog.state.query; onTextEdited: dialog.bridge.setParameter("query",text) }
            AppComboBox {
                colors: dialog.colors
                visible: dialog.state.workflow === "pl" && !dialog.state.savedOnly
                model: ["PL","REF","All"]
                currentIndex: model.indexOf(dialog.state.type)
                onActivated: dialog.bridge.setParameter("type",currentText)
                Accessible.name: "Source type"
            }
            AppComboBox {
                colors: dialog.colors
                visible: dialog.state.workflow === "pl" && !dialog.state.savedOnly
                model: ["All","New","Processed","History unknown"]
                property var values: ["all","unprocessed","processed","unknown"]
                currentIndex: values.indexOf(dialog.state.statusFilter)
                onActivated: dialog.bridge.setParameter("statusFilter",values[currentIndex])
                Accessible.name: "Processing status"
            }
            AppButton { colors: dialog.colors; text: "Refresh"; onClicked: dialog.bridge.invoke("refresh") }
        }
        Flow {
            Layout.fillWidth: true
            spacing: 5
            visible: !dialog.backgroundSelection
            Basic.CheckBox { text: "Saved DAT results"; checked: dialog.state.savedOnly; onToggled: dialog.bridge.setParameter("savedOnly",checked) }
            Basic.CheckBox { visible: dialog.state.workflow === "drr" && !dialog.state.savedOnly; text: "Unprocessed only"; checked: dialog.state.onlyNew; onToggled: dialog.bridge.setParameter("onlyNew",checked) }
            Basic.CheckBox { visible: dialog.state.workflow === "drr" && !dialog.state.savedOnly; text: "Include backgrounds"; checked: dialog.state.includeBackgrounds; onToggled: dialog.bridge.setParameter("includeBackgrounds",checked) }
            Basic.CheckBox { visible: dialog.state.workflow === "drr" && !dialog.state.savedOnly; text: "All experiment folders"; checked: dialog.state.includeAll; onToggled: dialog.bridge.setParameter("includeAll",checked) }
        }
        RowLayout {
            Layout.fillWidth: true
            visible: dialog.backgroundSelection
            Basic.Label { Layout.fillWidth: true; text: "Select files for this " + (dialog.state.sourceKindLabel || "") + " reference. Use selected updates the draft."; color: dialog.colors.muted; wrapMode: Text.Wrap }
            Basic.CheckBox { objectName: "source_catalog_show_all"; text: "Show all CSV files"; checked: dialog.state.showAllFiles === true; onToggled: dialog.bridge.setParameter("showAllFiles", checked) }
        }
        Basic.SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal
            CatalogListView {
                id: groups
                onMovementStarted: dialog.cancelReopenPosition()
                objectName: "source_catalog_groups"
                visible: !dialog.backgroundSelection && dialog.state.workflow === "drr" && !dialog.state.savedOnly
                Basic.SplitView.preferredWidth: visible ? 300 : 0
                Basic.SplitView.minimumWidth: visible ? 160 : 0
                model: dialog.bridge.groupsModel
                clip: true
                spacing: 8
                leftMargin: 3
                rightMargin: 7
                topMargin: 3
                bottomMargin: 3
                delegate: AppItemDelegate {
                    id: groupRow
                    colors: dialog.colors
                    required property var entry
                    objectName: "source_catalog_group_" + entry.id
                    width: groups.width - groups.leftMargin - groups.rightMargin
                    padding: 10
                    text: entry.name + "\n" + groupStatus.text + "\n" + entry.session + " · " + entry.detail
                    highlighted: entry.id === dialog.state.selectedGroup
                    onClicked: { dialog.cancelReopenPosition(); dialog.bridge.setParameter("selectedGroup",entry.id) }
                    onDoubleClicked: dialog.bridge.invoke("add-group")
                    Basic.ToolTip.visible: hovered
                    Basic.ToolTip.text: text
                    Basic.ToolTip.delay: 650
                    background: Rectangle {
                        radius: 5
                        color: groupRow.down ? dialog.colors.pressed : groupRow.highlighted ? dialog.colors.selected : groupRow.hovered ? dialog.colors.hover : dialog.colors.control
                        border.width: groupRow.visualFocus ? 2 : 1
                        border.color: groupRow.visualFocus ? dialog.colors.focus : groupRow.highlighted ? dialog.colors.accent : dialog.colors.border
                        Rectangle { anchors.left: parent.left; anchors.top: parent.top; anchors.bottom: parent.bottom; anchors.margins: 4; width: 3; radius: 1; visible: groupRow.highlighted; color: dialog.colors.accent }
                    }
                    contentItem: ColumnLayout {
                        spacing: 6
                        Text { Layout.fillWidth: true; text: groupRow.entry.name; color: dialog.colors.text; wrapMode: Text.Wrap; maximumLineCount: 3; elide: Text.ElideRight; font.pixelSize: 12; font.weight: Font.DemiBold }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 5
                            SourceStatusBadge {
                                id: groupStatus
                                objectName: "source_catalog_group_status_" + groupRow.entry.id
                                colors: dialog.colors
                                Layout.maximumWidth: groupRow.availableWidth - groupCount.implicitWidth - 5
                                status: groupRow.entry.processedCount >= groupRow.entry.count ? "processed" : groupRow.entry.processedCount > 0 ? "partial" : "new"
                                text: status === "partial" ? "Partial · " + groupRow.entry.processedCount + "/" + groupRow.entry.count : status === "processed" ? "Processed" : "Unprocessed"
                            }
                            Text { id: groupCount; Layout.fillWidth: true; text: groupRow.entry.count + (groupRow.entry.count === 1 ? " file" : " files"); horizontalAlignment: Text.AlignRight; color: dialog.colors.muted; font.pixelSize: 11 }
                        }
                        Text { Layout.fillWidth: true; text: groupRow.entry.session; color: dialog.colors.muted; font.pixelSize: 11 }
                        Text { Layout.fillWidth: true; text: groupRow.entry.detail; color: dialog.colors.muted; wrapMode: Text.Wrap; maximumLineCount: 2; elide: Text.ElideRight; font.pixelSize: 11 }
                    }
                }
            }
            CatalogListView {
                id: files
                onMovementStarted: dialog.cancelReopenPosition()
                objectName: "source_catalog_files"
                Basic.SplitView.fillWidth: true
                Basic.SplitView.minimumWidth: 250
                model: dialog.bridge.filesModel
                clip: true
                spacing: 5
                Basic.Label {
                    anchors.centerIn: parent
                    width: Math.max(100, parent.width - 32)
                    visible: dialog.backgroundSelection && files.count === 0 && !dialog.state.busy
                    text: "No matching references.\nTry another search, Show all CSV files, or Browse files…"
                    color: dialog.colors.muted
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                }
                delegate: CatalogFileRow {
                    id: fileRow
                    colors: dialog.colors
                    objectName: "source_catalog_file_" + entry.name
                    width: files.width
                    text: entry.name + "\n" + (dialog.backgroundSelection ? (entry.summary || entry.source) : entry.kind + " · " + entry.status.toUpperCase() + (entry.processedAt ? " · " + entry.processedAt.substring(0,16) : ""))
                    onSelectionRequested: selected => { dialog.cancelReopenPosition(); dialog.bridge.selectFile(entry.id, selected) }
                    Text { Layout.fillWidth: true; text: fileRow.entry.name; font: fileRow.font; color: dialog.colors.text; wrapMode: Text.Wrap; maximumLineCount: 3; elide: Text.ElideRight }
                    RowLayout {
                        Layout.fillWidth: true
                        visible: !dialog.backgroundSelection
                        SourceStatusBadge { objectName: "source_catalog_file_status_" + fileRow.entry.name; colors: dialog.colors; status: fileRow.entry.status; Layout.maximumWidth: fileRow.width - 56 }
                        Text { Layout.fillWidth: true; text: fileRow.entry.kind + (fileRow.entry.processedAt ? " · " + fileRow.entry.processedAt.substring(0,16) : ""); color: dialog.colors.muted; font.pixelSize: 11; elide: Text.ElideRight }
                    }
                    Text { Layout.fillWidth: true; visible: dialog.backgroundSelection; text: fileRow.entry.summary || fileRow.entry.source; color: dialog.colors.muted; wrapMode: Text.Wrap; maximumLineCount: 2; elide: Text.ElideRight; font.pixelSize: 11 }
                }
            }
            ColumnLayout {
                Basic.SplitView.preferredWidth: 250
                Basic.SplitView.minimumWidth: 170
                RowLayout {
                    Layout.fillWidth: true
                    Basic.Label { Layout.fillWidth: true; text: "Chosen files · " + dialog.state.selected.length; color: dialog.colors.text; font.bold: true }
                    AppButton {
                        objectName: "source_catalog_clear"
                        colors: dialog.colors
                        text: "Clear all"
                        compact: true
                        description: "Clear chosen files"
                        enabled: dialog.state.selected.length > 0
                        onClicked: dialog.bridge.invoke("clear")
                    }
                }
                CatalogListView {
                    id: chosen
                    objectName: "source_catalog_chosen"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: dialog.bridge.chosenModel
                    clip: true
                    Basic.Label {
                        anchors.centerIn: parent
                        width: Math.max(80, parent.width - 24)
                        visible: dialog.backgroundSelection && chosen.count === 0
                        text: "No files chosen"
                        color: dialog.colors.muted
                        horizontalAlignment: Text.AlignHCenter
                    }
                    delegate: AppItemDelegate {
                        id: chosenRow
                        objectName: "source_catalog_chosen_" + entry.name
                        colors: dialog.colors
                        required property var entry
                        width: chosen.width
                        focusPolicy: Qt.StrongFocus
                        text: entry.name
                        highlighted: entry.id === dialog.state.focused
                        onClicked: dialog.bridge.setParameter("focused",entry.id)
                        onDoubleClicked: dialog.bridge.selectFile(entry.id,false)
                        Keys.onPressed: event => {
                            if (event.key === Qt.Key_Delete || event.key === Qt.Key_Backspace) {
                                dialog.bridge.selectFile(entry.id, false)
                                event.accepted = true
                            }
                        }
                        Basic.ToolTip.visible: hovered
                        Basic.ToolTip.text: entry.source + "\nDouble-click or press Delete to remove from selection."
                        Basic.ToolTip.delay: 650
                        contentItem: RowLayout {
                            spacing: 4
                            Text { Layout.fillWidth: true; text: chosenRow.text; color: dialog.colors.text; font: chosenRow.font; elide: Text.ElideMiddle; wrapMode: Text.Wrap; maximumLineCount: 2 }
                            AppButton {
                                objectName: "source_catalog_remove_" + chosenRow.entry.name
                                colors: dialog.colors
                                text: "Remove"
                                compact: true
                                description: "Remove " + chosenRow.entry.name + " from selection"
                                onClicked: dialog.bridge.selectFile(chosenRow.entry.id, false)
                            }
                        }
                    }
                }
                Basic.Label { Layout.fillWidth: true; text: dialog.state.detail; color: dialog.colors.muted; wrapMode: Text.Wrap }
            }
        }
        Basic.ProgressBar { Layout.fillWidth: true; visible: dialog.state.busy; indeterminate: true }
        Basic.Label { Layout.fillWidth: true; text: dialog.state.error || dialog.state.status; color: dialog.state.error ? dialog.colors.error : dialog.colors.muted; wrapMode: Text.Wrap }
        RowLayout {
            objectName: "source_catalog_actions"
            Layout.fillWidth: true
            spacing: 6
            AppButton { objectName: "source_catalog_browse"; colors: dialog.colors; text: "Browse files…"; onClicked: dialog.bridge.invoke("browse") }
            AppButton {
                objectName: "source_catalog_selection_actions"
                colors: dialog.colors
                visible: !dialog.backgroundSelection && dialog.state.workflow === "drr" && !dialog.state.savedOnly
                text: "Selection actions…"
                description: "Add a measurement group or compatible repeats"
                onClicked: selectionMenu.open()
                Basic.Menu {
                    id: selectionMenu
                    objectName: "source_catalog_selection_menu"
                    y: parent.height
                    Basic.MenuItem {
                        objectName: "source_catalog_add_group"
                        text: "Add group"
                        enabled: dialog.state.canAddGroup
                        onTriggered: dialog.bridge.invoke("add-group")
                    }
                    Basic.MenuItem {
                        objectName: "source_catalog_add_compatible"
                        text: "Add compatible repeats"
                        enabled: dialog.state.canAddGroup
                        onTriggered: dialog.bridge.invoke("compatible")
                    }
                }
            }
            Item { Layout.fillWidth: true }
            AppButton { objectName: "source_catalog_load"; colors: dialog.colors; text: dialog.backgroundSelection ? dialog.state.acceptLabel : "Load selected"; primary: true; enabled: dialog.state.canAccept; onClicked: dialog.bridge.invoke("accept") }
            AppButton { objectName: "source_catalog_footer_close"; colors: dialog.colors; text: "Close"; onClicked: dialog.reject() }
        }
    }
}
