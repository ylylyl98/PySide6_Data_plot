pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: page
    objectName: "power_selection"
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    readonly property var draft: state.draft || ({})
    readonly property var selected: state.selectedMeasurement && state.selectedMeasurement.key ? state.selectedMeasurement : undefined
    property bool combineMode: state.combinationKeys.length > 0
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: combineMode ? "Choose same-channel power segments" : "Choose a power measurement group"

    function sourceLabel(key) {
        let source = state.availableSweeps.find(function(source) { return source.key === key })
        return source ? source.title : key ? "Missing source: " + key : "Unassigned"
    }
    function options() {
        let keys = selected ? selected.sources : []
        for (let role of ["single", "KK", "KKp"])
            if (draft[role] && keys.indexOf(draft[role]) < 0) keys = keys.concat([draft[role]])
        return [{value: "", label: "Unassigned"}].concat(keys.map(function(key) { return {value: key, label: page.sourceLabel(key)} }))
    }
    component RoleChoice: ColumnLayout {
        id: row
        required property string role
        spacing: 3
        Text { text: row.role === "single" ? "Single sweep" : row.role + " sweep"; color: page.colors.muted; wrapMode: Text.Wrap; Layout.fillWidth: true }
        AppComboBox {
            colors: page.colors
            id: choice
            objectName: "power_draft_" + row.role
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            readonly property string optionsSignature: JSON.stringify(page.options())
            readonly property var choices: JSON.parse(optionsSignature)
            model: choices.map(function(option) { return option.label })
            currentIndex: choices.findIndex(function(option) { return option.value === (page.draft[row.role] || "") })
            Accessible.name: row.role + " source sweep"
            Accessible.description: page.sourceLabel(page.draft[row.role] || "")
            hoverEnabled: true
            Basic.ToolTip.visible: hovered
            Basic.ToolTip.text: Accessible.description
            Basic.ToolTip.delay: 700
            onActivated: function(index) { page.bridge.setParameter("draft." + row.role, choices[index].value) }
        }
    }
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8
        Flow {
            Layout.fillWidth: true
            spacing: 6
            AppButton { objectName: "power_groups_mode"; colors: page.colors; text: "Measurement groups"; primary: !page.combineMode; onClicked: page.combineMode = false }
            AppButton { objectName: "power_combine_mode"; colors: page.colors; text: "Combine sweeps…"; primary: page.combineMode; onClicked: page.combineMode = true }
            AppButton { colors: page.colors; text: "Refresh"; enabled: !page.state.catalogBusy; onClicked: page.bridge.invoke("refresh") }
            AppButton { objectName: "power_cancel_selection"; colors: page.colors; text: "Return to plot"; enabled: Object.keys(page.state.activeSelection || {}).length > 0; onClicked: page.bridge.invoke("cancel-selection") }
        }
        Text {
            Layout.fillWidth: true
            text: page.combineMode ? "Add segments of the same channel. KK and KKp are combined separately; review corrections before saving."
                                   : "Choose a coherent measurement. Modification date controls ordering; channel and acquisition identity control pairing."
            color: page.colors.muted
            wrapMode: Text.Wrap
        }
        Text {
            Layout.fillWidth: true
            visible: page.state.selectionWarnings.length > 0
            text: page.state.selectionWarnings.join("\n")
            color: page.colors.error
            wrapMode: Text.Wrap
        }
        GridLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            columns: page.width < 760 ? 1 : 2
            columnSpacing: 12
            rowSpacing: 8
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: 180
                Layout.preferredWidth: 360
                Layout.preferredHeight: page.width < 760 ? 240 : -1
                Layout.minimumHeight: 150
                color: page.colors.control
                border.color: page.colors.border
                radius: 4
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 6
                    Text { text: page.combineMode ? "Available sweeps" : "Recent measurement groups"; color: page.colors.text; font.weight: Font.DemiBold; Layout.fillWidth: true }
                    ListView {
                        id: catalogList
                        objectName: "power_source_list"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 2
                        model: page.combineMode ? page.state.availableSweeps.filter(function(source) {
                            return (!page.state.query || (source.title + " " + source.key).toLowerCase().indexOf(page.state.query.toLowerCase()) >= 0)
                                && (page.state.historyFilter === "all" || source.status.toLowerCase() === page.state.historyFilter)
                        }) : page.state.measurementGroups
                        delegate: Basic.ItemDelegate {
                            id: entry
                            required property var modelData
                            required property int index
                            width: catalogList.width
                            highlighted: page.combineMode ? modelData.chosen : modelData.key === page.state.selectedGroup
                            implicitHeight: entryText.implicitHeight + 16
                            Accessible.name: page.combineMode ? modelData.title : modelData.label
                            Accessible.description: page.combineMode ? modelData.key : modelData.sources.join("\n")
                            contentItem: Column {
                                id: entryText
                                spacing: 3
                                Text { width: parent.width; text: page.combineMode ? entry.modelData.title : entry.modelData.label; color: page.colors.text; wrapMode: Text.Wrap; font.weight: entry.highlighted ? Font.DemiBold : Font.Normal }
                                Text { width: parent.width; text: page.combineMode ? entry.modelData.status + (entry.modelData.combined ? " · Combined" : "") : entry.modelData.modifiedText; color: page.colors.muted; wrapMode: Text.Wrap; font.pixelSize: 11 }
                            }
                            background: Rectangle { color: entry.highlighted ? page.colors.selected : entry.hovered ? page.colors.hover : "transparent"; radius: 3 }
                            Basic.ToolTip.visible: hovered
                            Basic.ToolTip.text: Accessible.description
                            Basic.ToolTip.delay: 700
                            onClicked: {
                                catalogList.currentIndex = index
                                if (page.combineMode) page.bridge.invoke("toggle-combine:" + modelData.key)
                                else page.bridge.setParameter("selectedGroup", modelData.key)
                            }
                        }
                        Basic.ScrollBar.vertical: Basic.ScrollBar {}
                    }
                    AppButton { visible: !page.combineMode && page.state.olderCount > 0; colors: page.colors; text: "Show older measurements (" + page.state.olderCount + ")"; onClicked: page.bridge.invoke("show-older") }
                    Text { visible: catalogList.count === 0; Layout.fillWidth: true; text: page.state.catalogBusy ? "Scanning measurements…" : "No matching power sweeps. Select CSV files or enable filename-based series."; color: page.colors.muted; wrapMode: Text.Wrap }
                }
            }
            Basic.ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: 180
                Layout.preferredWidth: 360
                contentWidth: availableWidth
                ColumnLayout {
                    width: parent.width
                    spacing: 8
                    Text { Layout.fillWidth: true; text: page.combineMode ? "Chosen segments" : "Selected measurement"; color: page.colors.text; font.weight: Font.DemiBold }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: !page.combineMode
                        Text { Layout.fillWidth: true; text: page.selected ? page.selected.label : "Choose a measurement group."; color: page.colors.muted; wrapMode: Text.Wrap }
                        Text { Layout.fillWidth: true; visible: !!page.selected && page.selected.issues.length > 0; text: page.selected ? page.selected.issues.join("\n") : ""; color: page.colors.error; wrapMode: Text.Wrap }
                        RoleChoice { Layout.fillWidth: true; role: "single" }
                        AppButton { objectName: "power_open_single"; colors: page.colors; text: "Open sweep"; primary: true; enabled: !!page.draft.single && !page.state.busy; onClicked: page.bridge.invoke("open-sweep") }
                        Basic.CheckBox { id: details; text: "Details / Adjust pairing"; checked: !!page.selected && (Object.keys(page.selected.duplicates).length > 0 || page.selected.issues.length > 0); Accessible.name: text }
                        ColumnLayout {
                            Layout.fillWidth: true
                            visible: details.checked
                            RoleChoice { Layout.fillWidth: true; role: "KK" }
                            RoleChoice { Layout.fillWidth: true; role: "KKp" }
                            AppButton { colors: page.colors; text: "Swap KK ↔ KKp"; onClicked: page.bridge.invoke("swap-roles") }
                            AppComboBox {
                                colors: page.colors
                                Layout.fillWidth: true
                                Layout.minimumWidth: 0
                                property var values: ["auto", "stage", "power"]
                                model: ["Automatic VP pairing", "Pair by Stage", "Power interpolation"]
                                currentIndex: values.indexOf(page.draft.pairing || "auto")
                                Accessible.name: "VP pairing method"
                                onActivated: function(index) { page.bridge.setParameter("draft.pairing", values[index]) }
                            }
                        }
                        Text { Layout.fillWidth: true; text: "KK: " + page.sourceLabel(page.draft.KK || "") + "\nKKp: " + page.sourceLabel(page.draft.KKp || ""); color: page.colors.muted; wrapMode: Text.Wrap }
                        Flow {
                            Layout.fillWidth: true
                            spacing: 5
                            AppButton { objectName: "power_open_comparison"; colors: page.colors; text: "Open comparison"; enabled: !!page.draft.KK && !!page.draft.KKp && page.draft.KK !== page.draft.KKp && !page.state.busy; onClicked: page.bridge.invoke("open-comparison") }
                            AppButton { objectName: "power_open_vp"; colors: page.colors; text: "Open VP"; enabled: !!page.draft.KK && !!page.draft.KKp && page.draft.KK !== page.draft.KKp && !page.state.busy; onClicked: page.bridge.invoke("open-vp") }
                        }
                        Text { Layout.fillWidth: true; text: "Opening validates the selected files afresh. Confirmed manual pairing is portable in .power-selection.json; returning to the plot leaves it unchanged."; color: page.colors.muted; wrapMode: Text.Wrap; font.pixelSize: 11 }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: page.combineMode
                        Repeater {
                            model: page.state.combinationKeys
                            delegate: RowLayout {
                                id: chosen
                                required property var modelData
                                Layout.fillWidth: true
                                Text { text: page.sourceLabel(chosen.modelData); color: page.colors.text; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.minimumWidth: 0 }
                                AppButton { colors: page.colors; text: "Remove"; compact: true; onClicked: page.bridge.invoke("toggle-combine:" + chosen.modelData) }
                            }
                        }
                        Flow {
                            Layout.fillWidth: true
                            spacing: 5
                            AppButton { colors: page.colors; text: "Clear"; onClicked: page.bridge.invoke("clear-combination") }
                            AppButton { objectName: "power_review_combination"; colors: page.colors; text: "Review combination…"; primary: true; enabled: page.state.combinationKeys.length >= 2 && !page.state.busy; onClicked: page.bridge.invoke("review-combination") }
                        }
                        Text { Layout.fillWidth: true; text: page.state.combinationKeys.length + " sweep(s) chosen. Search/status filters retain this queue and your existing review settings."; color: page.colors.muted; wrapMode: Text.Wrap }
                    }
                }
            }
        }
        Text { Layout.fillWidth: true; text: page.state.status; color: page.state.error ? page.colors.error : page.colors.muted; wrapMode: Text.Wrap }
    }
}
