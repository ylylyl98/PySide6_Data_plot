pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Dialogs
import QtQuick.Layouts
import "DialogState.js" as DialogState

Rectangle {
    id: page
    objectName: "quick_presentation_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    readonly property bool compact: width < 1050
    color: colors.surface
    Accessible.role: Accessible.Pane
    Accessible.name: "PowerPoint presentation builder"

    function chooseSource() {
        DialogState.prepareFile(sourceDialog, state.sourceFolderUrl, "")
        sourceDialog.open()
    }
    function chooseOutput() {
        DialogState.prepareFile(outputDialog, state.outputFolderUrl, state.outputFileUrl)
        outputDialog.open()
    }
    function choosePlotFolder() {
        DialogState.prepareFolder(plotDialog, state.imageFolderUrl)
        plotDialog.open()
    }

    function selectRow(list, section, key, modifiers) {
        let target = key === Qt.Key_Home ? 0 : key === Qt.Key_End ? list.count - 1
                   : Math.max(0, Math.min(list.count - 1, list.currentIndex + (key === Qt.Key_Down ? 1 : -1)))
        list.currentIndex = target
        if (section === "available") bridge.selectAvailableRow(target, modifiers)
        else bridge.selectQueueRow(target, modifiers)
        list.positionViewAtIndex(target, ListView.Contain)
    }

    component Caption: Text {
        color: page.colors.muted
        font.pixelSize: 11
        verticalAlignment: Text.AlignVCenter
    }
    component PathEdit: Basic.TextField {
        property string optionKey
        readonly property string committedValue: page.state[optionKey] || ""
        implicitHeight: 30
        Layout.fillWidth: true
        Layout.minimumWidth: 60
        selectByMouse: true
        enabled: !page.state.building
        onCommittedValueChanged: if (!activeFocus) text = committedValue
        Component.onCompleted: text = committedValue
        onEditingFinished: page.bridge.setOption(optionKey, text)
    }
    component SectionTitle: Text {
        Layout.fillWidth: true
        color: page.colors.text
        font.weight: Font.DemiBold
        elide: Text.ElideRight
    }

    AppFileDialog {
        colors: page.colors
        parentWindow: page.Window.window
        id: sourceDialog
        objectName: "slides_source_dialog"
        title: "Choose existing PowerPoint"
        nameFilters: ["PowerPoint (*.pptx)"]
        onAccepted: page.bridge.setOption("source", selectedFile.toString())
    }
    AppFileDialog {
        colors: page.colors
        parentWindow: page.Window.window
        id: outputDialog
        objectName: "slides_output_dialog"
        title: "Save PowerPoint copy"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "pptx"
        nameFilters: ["PowerPoint (*.pptx)"]
        onAccepted: page.bridge.setOption("output", selectedFile.toString())
    }
    AppFolderDialog {
        colors: page.colors
        parentWindow: page.Window.window
        id: plotDialog
        objectName: "slides_plot_folder_dialog"
        title: "Choose processed plot folder"
        onAccepted: page.bridge.setOption("imageRoot", selectedFolder.toString())
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 6
        Text {
            Layout.fillWidth: true
            text: "Insert processed plots into PowerPoint"
            color: page.colors.text
            font.pixelSize: 17
            font.weight: Font.DemiBold
            Accessible.role: Accessible.Heading
        }
        Caption {
            Layout.fillWidth: true
            text: "Choose a deck, select PNG plots, and arrange the queue. Images keep their original aspect ratio."
            wrapMode: Text.WordWrap
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 5
            Caption { text: "Presentation"; Layout.preferredWidth: 78 }
            PathEdit { objectName: "slides_source"; optionKey: "source"; placeholderText: "Existing .pptx to edit"; Accessible.name: "Existing PowerPoint presentation" }
            AppButton { objectName: "slides_browse_source"; colors: page.colors; text: "Browse…"; enabled: !page.state.building; description: "Choose the existing PowerPoint presentation"; onClicked: page.chooseSource() }
            AppButton { objectName: "slides_advanced"; colors: page.colors; text: "Advanced"; description: "Show optional copy path and recovery backup"; checkable: true; checked: page.state.advanced; onClicked: page.bridge.setOption("advanced", !page.state.advanced) }
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 5
            Caption { text: "Plot folder"; Layout.preferredWidth: 78 }
            PathEdit { objectName: "slides_image_root"; optionKey: "imageRoot"; placeholderText: "Processed Data folder containing PNG plots"; Accessible.name: "Processed plot folder" }
            AppButton { objectName: "slides_browse_root"; colors: page.colors; text: "Browse…"; enabled: !page.state.building; onClicked: page.choosePlotFolder() }
            AppButton { objectName: "slides_refresh"; colors: page.colors; text: page.state.scanning ? "Cancel scan" : "Refresh"; enabled: !page.state.building; onClicked: page.state.scanning ? page.bridge.cancel() : page.bridge.refresh() }
        }
        RowLayout {
            Layout.fillWidth: true
            visible: page.state.advanced
            spacing: 5
            Caption { text: "Optional copy"; Layout.preferredWidth: 78 }
            PathEdit { objectName: "slides_output"; optionKey: "output"; placeholderText: "Create a separate copy instead"; Accessible.name: "Optional PowerPoint copy path" }
            AppButton { objectName: "slides_browse_output"; colors: page.colors; text: "Save as…"; enabled: !page.state.building; onClicked: page.chooseOutput() }
            Basic.CheckBox { objectName: "slides_backup"; text: "Recovery backup"; checked: page.state.backup; enabled: !page.state.building; onToggled: page.bridge.setOption("backup", checked) }
        }
        Flow {
            Layout.fillWidth: true
            spacing: 8
            Row {
                spacing: 5
                Caption { text: "Per slide"; height: 30 }
                AppComboBox {
                    colors: page.colors
                    objectName: "slides_count"
                    width: 140; height: 30
                    model: ["Auto (up to 12)", "1 image", "2 images", "3 images", "4 images", "5 images", "6 images", "7 images", "8 images", "9 images", "10 images", "11 images", "12 images"]
                    currentIndex: page.state.imagesPerSlide
                    enabled: !page.state.building
                    Accessible.name: "Images per slide"
                    onActivated: function(index) { page.bridge.setOption("imagesPerSlide", index) }
                }
            }
            Row {
                spacing: 5
                Caption { text: "Group"; height: 30 }
                AppComboBox {
                    colors: page.colors
                    objectName: "slides_group"
                    width: 145; height: 30
                    property var values: ["doping", "folder", "gate", "efield", "b_range", "queue"]
                    model: ["Doping", "Measurement folder", "Gate voltage", "E-field", "B-field range", "Queue order"]
                    currentIndex: values.indexOf(page.state.groupBy)
                    enabled: !page.state.building
                    Accessible.name: "Group slides by"
                    Accessible.description: "Grouping uses only queued PNGs and never adds unselected energies."
                    onActivated: function(index) { page.bridge.setOption("groupBy", values[index]) }
                }
            }
            Row {
                spacing: 5
                Caption { text: "Captions"; height: 30 }
                AppComboBox {
                    colors: page.colors
                    objectName: "slides_caption"
                    width: 105; height: 30
                    property var values: ["none", "minimal", "full"]
                    model: ["None", "Minimal", "Full filename"]
                    currentIndex: values.indexOf(page.state.caption)
                    enabled: !page.state.building
                    Accessible.name: "Image captions"
                    onActivated: function(index) { page.bridge.setOption("caption", values[index]) }
                }
            }
            Basic.CheckBox { objectName: "slides_labels"; height: 30; text: "Panel labels"; checked: page.state.panelLabels; enabled: !page.state.building; onToggled: page.bridge.setOption("panelLabels", checked) }
            Basic.TextField { objectName: "slides_title"; width: Math.max(150, Math.min(260, page.width - 20)); height: 30; text: page.state.title; placeholderText: "Optional slide title"; selectByMouse: true; enabled: !page.state.building; Accessible.name: "Slide title"; onEditingFinished: page.bridge.setOption("title", text) }
        }

        Basic.SplitView {
            id: splitter
            objectName: "slides_splitter"
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal
            onResizingChanged: if (!resizing) page.bridge.setSplitterSizes([availablePane.width, queuePane.width, previewPane.width])
            handle: Rectangle { implicitWidth: 5; color: Basic.SplitHandle.hovered || Basic.SplitHandle.pressed ? page.colors.border : page.colors.divider }

            Rectangle {
                id: availablePane
                objectName: "slides_available_pane"
                Basic.SplitView.preferredWidth: page.state.splitterSizes[0]
                Basic.SplitView.minimumWidth: 190
                color: page.colors.control
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 5
                    SectionTitle { text: "Available processed plots · " + page.state.availableCount }
                    Basic.TextField { objectName: "slides_search"; Layout.fillWidth: true; implicitHeight: 30; text: page.state.query; placeholderText: "Search filename or folder…"; selectByMouse: true; Accessible.name: "Search processed plots"; onTextEdited: page.bridge.setOption("query", text) }
                    AppComboBox {
                        colors: page.colors
                        objectName: "slides_workflow_filter"
                        Layout.fillWidth: true; implicitHeight: 28
                        model: ["All folders"].concat(page.state.workflows)
                        currentIndex: page.state.workflow ? page.state.workflows.indexOf(page.state.workflow) + 1 : 0
                        Accessible.name: "Filter by workflow folder"
                        onActivated: function(index) { page.bridge.setOption("workflow", index === 0 ? "" : page.state.workflows[index - 1]) }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        AppComboBox {
                            colors: page.colors
                            objectName: "slides_kind_filter"
                            Layout.fillWidth: true; Layout.minimumWidth: 60; implicitHeight: 28
                            property var values: ["", "mcd_combo", "mcd_b"]
                            model: ["All types", "MCD Combo", "MCD(B)"]
                            currentIndex: values.indexOf(page.state.plotKind)
                            Accessible.name: "Filter by plot type"
                            onActivated: function(index) { page.bridge.setOption("plotKind", values[index]) }
                        }
                        AppComboBox {
                            colors: page.colors
                            objectName: "slides_sort"
                            Layout.fillWidth: true; Layout.minimumWidth: 60; implicitHeight: 28
                            property var values: ["newest", "oldest", "filename"]
                            model: ["Newest first", "Oldest first", "Filename A–Z"]
                            currentIndex: values.indexOf(page.state.sort)
                            Accessible.name: "Sort available plots"
                            onActivated: function(index) { page.bridge.setOption("sort", values[index]) }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Caption {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: {
                                let selected = page.state.lastSelectedIndex
                                let first = availableList.indexAt(1, availableList.contentY + 1)
                                let last = availableList.indexAt(1, availableList.contentY + availableList.height - 1)
                                if (selected < 0) return "No PNG selected"
                                return page.state.selectedAvailableCount + " selected · " + (first >= 0 && selected < first ? "above" : last >= 0 && selected > last ? "below" : "visible")
                            }
                        }
                        AppButton { objectName: "slides_show_selected"; colors: page.colors; text: "Show"; description: "Show the most recently selected PNG"; enabled: page.state.lastSelectedIndex >= 0; onClicked: availableList.positionViewAtIndex(page.state.lastSelectedIndex, ListView.Center) }
                    }
                    ListView {
                        id: availableList
                        objectName: "slides_available_list"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        reuseItems: true
                        model: page.bridge.availableModel
                        focus: true
                        currentIndex: -1
                        boundsBehavior: Flickable.StopAtBounds
                        Basic.ScrollBar.vertical: Basic.ScrollBar {}
                        Accessible.name: "Available processed PNG plots"
                        Keys.onPressed: function(event) {
                            if ([Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End].indexOf(event.key) >= 0) {
                                page.selectRow(availableList, "available", event.key, event.modifiers); event.accepted = true
                            } else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) {
                                page.bridge.addSelected(); event.accepted = true
                            } else if (event.key === Qt.Key_A && event.modifiers & Qt.ControlModifier) {
                                page.bridge.selectAll("available"); event.accepted = true
                            }
                        }
                        delegate: Rectangle {
                            id: availableRow
                            required property var model
                            required property int index
                            objectName: "slides_available_row_" + index
                            width: availableList.width
                            height: Math.max(76, rowText.implicitHeight + 12)
                            color: model.selected ? page.colors.selected : index % 2 ? page.colors.surface : page.colors.control
                            border.width: model.selected ? 1 : 0
                            border.color: page.colors.selectedBorder
                            Accessible.role: Accessible.ListItem
                            Accessible.name: model.name
                            Accessible.description: model.path
                            Accessible.selected: model.selected
                            Image { id: thumbnail; objectName: "slides_thumbnail_" + availableRow.index; readonly property bool imageReady: status === Image.Ready; x: 4; y: 6; width: Math.min(70, availableRow.width * .25); height: 54; source: availableRow.model.url; sourceSize: Qt.size(140, 108); fillMode: Image.PreserveAspectFit; asynchronous: true; Accessible.ignored: true }
                            Column {
                                id: rowText
                                x: thumbnail.width + 10; y: 6
                                width: availableRow.width - x - 8
                                spacing: 3
                                Text { width: parent.width; text: availableRow.model.name; color: page.colors.text; wrapMode: Text.WrapAnywhere; font.pixelSize: 12 }
                                Caption { width: parent.width; text: availableRow.model.badge + " · Folder: " + availableRow.model.folder; wrapMode: Text.WrapAnywhere }
                                Caption { width: parent.width; text: availableRow.model.modified; wrapMode: Text.WrapAnywhere }
                            }
                            MouseArea {
                                anchors.fill: parent
                                acceptedButtons: Qt.LeftButton
                                onClicked: function(mouse) { availableList.currentIndex = availableRow.index; availableList.forceActiveFocus(); page.bridge.selectAvailableRow(availableRow.index, mouse.modifiers) }
                                onDoubleClicked: page.bridge.addSelected()
                            }
                        }
                        Text { anchors.centerIn: parent; width: parent.width - 12; text: page.state.scanning ? "Scanning PNG plots…" : "No matching processed PNG plots"; visible: availableList.count === 0; color: page.colors.muted; wrapMode: Text.WordWrap; horizontalAlignment: Text.AlignHCenter }
                    }
                    Flow {
                        Layout.fillWidth: true
                        spacing: 3
                        AppButton { objectName: "slides_add_selected"; colors: page.colors; text: "Add selected →"; enabled: !page.state.building && page.state.selectedAvailableCount > 0; onClicked: page.bridge.addSelected() }
                        AppButton { objectName: "slides_add_all"; colors: page.colors; text: "Add all shown"; enabled: !page.state.building && page.state.availableCount > 0; onClicked: page.bridge.addAll() }
                        AppButton { objectName: "slides_add_mcd_pair"; colors: page.colors; text: "Newest MCD pair"; enabled: !page.state.building; description: "Add the newest matching MCD Combo and MCD(B) pair from the same folder"; onClicked: page.bridge.addNewestMcdPair() }
                    }
                }
            }
            Rectangle {
                id: queuePane
                objectName: "slides_queue_pane"
                Basic.SplitView.preferredWidth: page.state.splitterSizes[1]
                Basic.SplitView.minimumWidth: 170
                color: page.colors.control
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 5
                    SectionTitle { text: "Slide image order · " + page.state.queueCount }
                    Caption { Layout.fillWidth: true; text: "Drag to reorder. Ctrl/Shift selects multiple plots."; wrapMode: Text.WordWrap }
                    Caption { Layout.fillWidth: true; visible: text.length > 0; text: page.state.mcdStatus; wrapMode: Text.WordWrap }
                    ListView {
                        id: queueList
                        objectName: "slides_queue_list"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        reuseItems: true
                        currentIndex: -1
                        model: page.bridge.queueModel
                        boundsBehavior: Flickable.StopAtBounds
                        Basic.ScrollBar.vertical: Basic.ScrollBar {}
                        Accessible.name: "Queued slide images in presentation order"
                        Keys.onPressed: function(event) {
                            if (event.key === Qt.Key_Delete || event.key === Qt.Key_Backspace) {
                                page.bridge.removeSelected(); event.accepted = true
                            } else if (event.key === Qt.Key_A && event.modifiers & Qt.ControlModifier) {
                                page.bridge.selectAll("queue"); event.accepted = true
                            } else if ([Qt.Key_Up, Qt.Key_Down].indexOf(event.key) >= 0 && event.modifiers & Qt.AltModifier) {
                                page.bridge.moveSelected(event.key === Qt.Key_Up ? -1 : 1); event.accepted = true
                            } else if ([Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End].indexOf(event.key) >= 0) {
                                page.selectRow(queueList, "queue", event.key, event.modifiers); event.accepted = true
                            }
                        }
                        delegate: Item {
                            id: queueRow
                            required property var model
                            required property int index
                            objectName: "slides_queue_row_" + index
                            width: queueList.width
                            height: Math.max(60, queueText.implicitHeight + 12)
                            Accessible.role: Accessible.ListItem
                            Accessible.name: model.position + ". " + model.name
                            Accessible.selected: model.selected
                            Rectangle {
                                id: movable
                                width: parent.width
                                height: parent.height
                                z: dragArea.drag.active ? 10 : 0
                                color: queueRow.model.selected ? page.colors.selected : queueRow.index % 2 ? page.colors.surface : page.colors.control
                                border.width: queueRow.model.selected ? 1 : 0
                                border.color: page.colors.selectedBorder
                                Column {
                                    id: queueText
                                    anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
                                    anchors.margins: 6
                                    spacing: 4
                                    Text { width: parent.width; text: queueRow.model.position + ". " + queueRow.model.name; color: page.colors.text; wrapMode: Text.WrapAnywhere; font.pixelSize: 12 }
                                    Caption { width: parent.width; text: queueRow.model.badge + " · Folder: " + queueRow.model.folder; wrapMode: Text.WrapAnywhere }
                                }
                                MouseArea {
                                    id: dragArea
                                    anchors.fill: parent
                                    drag.target: page.state.building ? null : movable
                                    drag.axis: Drag.YAxis
                                    preventStealing: drag.active
                                    onClicked: function(mouse) { queueList.currentIndex = queueRow.index; queueList.forceActiveFocus(); page.bridge.selectQueueRow(queueRow.index, mouse.modifiers) }
                                    onReleased: function(mouse) {
                                        if (drag.active) {
                                            let position = mapToItem(queueList.contentItem, mouse.x, mouse.y)
                                            let target = queueList.indexAt(position.x, position.y)
                                            if (target >= 0) page.bridge.moveQueue(queueRow.index, target)
                                        }
                                        movable.y = 0
                                    }
                                    onCanceled: movable.y = 0
                                }
                            }
                        }
                    }
                    Flow {
                        Layout.fillWidth: true
                        spacing: 3
                        AppButton { objectName: "slides_up"; colors: page.colors; text: "Up"; enabled: !page.state.building && page.state.selectedQueueCount > 0; description: "Move selected plots up (Alt+Up)"; onClicked: page.bridge.moveSelected(-1) }
                        AppButton { objectName: "slides_down"; colors: page.colors; text: "Down"; enabled: !page.state.building && page.state.selectedQueueCount > 0; description: "Move selected plots down (Alt+Down)"; onClicked: page.bridge.moveSelected(1) }
                        AppButton { objectName: "slides_remove"; colors: page.colors; text: "Remove"; enabled: !page.state.building && page.state.selectedQueueCount > 0; onClicked: page.bridge.removeSelected() }
                        AppButton { objectName: "slides_clear"; colors: page.colors; text: "Clear"; enabled: !page.state.building && page.state.queueCount > 0; onClicked: page.bridge.clearQueue() }
                    }
                }
            }
            Rectangle {
                id: previewPane
                objectName: "slides_preview_pane"
                Basic.SplitView.preferredWidth: page.state.splitterSizes[2]
                Basic.SplitView.minimumWidth: 190
                Basic.SplitView.fillWidth: true
                color: page.colors.control
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 6
                    SectionTitle { text: "Slide preview" }
                    AppComboBox {
                        colors: page.colors
                        objectName: "slides_preview_selector"
                        Layout.fillWidth: true
                        implicitHeight: 30
                        model: page.state.slides.map(function(slide) { return slide.label })
                        currentIndex: page.state.previewIndex
                        enabled: page.state.slides.length > 0
                        Accessible.name: "Preview slide"
                        onActivated: function(index) { page.bridge.setPreviewSlide(index) }
                    }
                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Rectangle {
                            id: previewFrame
                            objectName: "slides_preview_frame"
                            anchors.centerIn: parent
                            width: Math.min(parent.width, parent.height * 4 / 3)
                            height: width * 3 / 4
                            color: "#ffffff"
                            border.color: page.colors.border
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 5
                                Text { Layout.fillWidth: true; text: page.state.preview.title || ""; visible: text.length > 0; color: "#1D293B"; font.pixelSize: 12; font.weight: Font.DemiBold; wrapMode: Text.WordWrap }
                                GridLayout {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    columns: page.state.preview.columns || 1
                                    rows: page.state.preview.rows || 1
                                    columnSpacing: 4
                                    rowSpacing: 4
                                    Repeater {
                                        model: page.state.preview.images || []
                                        delegate: Item {
                                            id: previewCell
                                            required property var modelData
                                            required property int index
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            Layout.preferredWidth: 100
                                            Layout.preferredHeight: 100
                                            ColumnLayout {
                                                anchors.fill: parent
                                                spacing: 2
                                                Image {
                                                    objectName: "slides_preview_image_" + previewCell.index
                                                    readonly property bool imageReady: status === Image.Ready
                                                    Layout.fillWidth: true
                                                    Layout.fillHeight: true
                                                    Layout.minimumHeight: 1
                                                    source: previewCell.modelData.url
                                                    // Decode against the independent slide frame. The
                                                    // image's implicit size must not feed its own layout.
                                                    sourceSize: Qt.size(Math.max(32, Math.floor(previewFrame.width / (page.state.preview.columns || 1) * 2)),
                                                                        Math.max(32, Math.floor(previewFrame.height / (page.state.preview.rows || 1) * 2)))
                                                    fillMode: Image.PreserveAspectFit
                                                    asynchronous: true
                                                    Accessible.name: previewCell.modelData.name
                                                    Text { anchors.centerIn: parent; text: "Image unavailable"; visible: parent.status === Image.Error; color: "#1D293B"; font.pixelSize: 10; width: parent.width; wrapMode: Text.WordWrap; horizontalAlignment: Text.AlignHCenter }
                                                    Text { x: 2; y: 2; text: previewCell.modelData.panelLabel; color: "#1D293B"; font.pixelSize: 10; font.weight: Font.Bold; visible: text.length > 0 }
                                                }
                                                Text { Layout.fillWidth: true; text: previewCell.modelData.caption; visible: text.length > 0; color: "#46566D"; font.pixelSize: 9; wrapMode: Text.WrapAnywhere; horizontalAlignment: Text.AlignHCenter }
                                            }
                                        }
                                    }
                                }
                            }
                            Text { anchors.centerIn: parent; width: parent.width - 20; text: "Add plots to preview the slide layout"; visible: !(page.state.preview.images || []).length; color: "#46566D"; wrapMode: Text.WordWrap; horizontalAlignment: Text.AlignHCenter; font.pixelSize: 11 }
                        }
                    }
                    Caption { Layout.fillWidth: true; text: "Preview is approximate. PowerPoint preserves each PNG’s aspect ratio; captions remain editable text."; wrapMode: Text.WordWrap }
                }
            }
        }
        Text {
            objectName: "slides_status"
            Layout.fillWidth: true
            text: page.state.status
            color: page.state.error ? page.colors.error : page.colors.muted
            wrapMode: Text.WordWrap
            font.pixelSize: 11
            Accessible.name: text
        }
        Caption {
            Layout.fillWidth: true
            visible: !page.state.liveAvailable
            text: page.state.liveReason
            wrapMode: Text.WordWrap
        }
        Flow {
            Layout.fillWidth: true
            spacing: 4
            Caption { text: page.state.summary; height: 30; width: Math.min(210, implicitWidth + 12) }
            Basic.BusyIndicator { width: 28; height: 28; running: page.state.building || page.state.scanning; visible: running }
            AppButton { objectName: "slides_cancel"; colors: page.colors; text: "Cancel"; visible: page.state.cancellable; onClicked: page.bridge.cancel() }
            AppButton { objectName: "slides_open_output"; colors: page.colors; text: "Open presentation"; enabled: page.state.canOpenOutput; onClicked: page.bridge.openOutput() }
            AppButton { objectName: "slides_build_copy"; colors: page.colors; text: "Save a copy"; enabled: !page.state.building; onClicked: page.bridge.build("copy") }
            AppButton { objectName: "slides_build_live"; colors: page.colors; text: "Insert live"; enabled: !page.state.building && page.state.liveAvailable; description: page.state.liveAvailable ? "Insert into the selected open PowerPoint without saving it" : page.state.liveReason; onClicked: page.bridge.build("live") }
            AppButton { objectName: "slides_build_save"; colors: page.colors; text: "Insert and save"; primary: true; enabled: !page.state.building; onClicked: page.bridge.build("auto_save") }
        }
    }
}
