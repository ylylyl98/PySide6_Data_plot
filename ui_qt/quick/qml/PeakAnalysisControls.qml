pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: panel
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    color: colors.surface
    objectName: "peak_analysis_controls"
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Basic.Label {
            Layout.fillWidth: true; Layout.margins: 10
            visible: !panel.state.hasDataset
            text: "Analysis uses immutable numeric snapshots. Saved workspaces retain all previous versions and results."
            wrapMode: Text.WordWrap; color: panel.colors.muted
        }
        ListView {
            id: branches
            Layout.fillWidth: true; Layout.leftMargin: 8; Layout.rightMargin: 8
            Layout.preferredHeight: visible ? Math.min(210, contentHeight) : 0
            visible: panel.state.controlsPage === "branches" && panel.state.hasDataset
            model: panel.bridge.branchesModel
            clip: true
            Basic.ScrollBar.vertical: Basic.ScrollBar { }
            delegate: Rectangle {
                id: branchRow
                required property var entry
                width: branches.width; height: 38
                color: panel.state.selectedBranch === entry.id ? panel.colors.selected : "transparent"
                RowLayout {
                    anchors.fill: parent; anchors.margins: 3; spacing: 4
                    Basic.CheckBox {
                        checked: branchRow.entry.enabled !== false
                        enabled: !panel.state.busy
                        Accessible.name: "Include " + branchRow.entry.name
                        onToggled: panel.bridge.setParameter("branch." + branchRow.entry.id + ".enabled", checked)
                    }
                    Rectangle { width: 8; height: 20; radius: 3; color: branchRow.entry.displayColor }
                    Basic.TextField {
                        id: nameEditor
                        Layout.fillWidth: true; Layout.minimumWidth: 0
                        text: branchRow.entry.name
                        enabled: !panel.state.busy
                        selectByMouse: true
                        Accessible.name: "Branch name " + branchRow.entry.name
                        onActiveFocusChanged: if (activeFocus) panel.bridge.setParameter("selectedBranch", branchRow.entry.id)
                        onEditingFinished: if (text !== branchRow.entry.name) panel.bridge.setParameter("branch." + branchRow.entry.id + ".name", text)
                    }
                    AppButton {
                        text: "Inspect"; compact: true; colors: panel.colors
                        enabled: !panel.state.busy
                        onClicked: panel.bridge.setParameter("selectedBranch", branchRow.entry.id)
                    }
                }
            }
        }
        AppParameterPane {
            Layout.fillWidth: true; Layout.fillHeight: true
            bridge: panel.bridge; colors: panel.colors
            groups: panel.state.parameterGroups || []
        }
        RowLayout {
            Layout.fillWidth: true; Layout.margins: 8
            spacing: 6
            visible: panel.state.hasDataset && panel.state.controlsPage !== "branches"
            AppButton {
                Layout.fillWidth: true
                objectName: "app_parameter_find-fit"
                colors: panel.colors; text: "Find & Fit"; primary: true
                visible: panel.state.kind === "PL"
                enabled: panel.state.canAnalyze
                onClicked: panel.bridge.invoke("find-fit")
            }
            AppButton {
                Layout.fillWidth: true
                objectName: "app_parameter_detect"
                colors: panel.colors
                text: panel.state.kind === "PL" ? "Find only" : "Find"
                primary: panel.state.kind !== "PL"
                enabled: panel.state.canAnalyze
                onClicked: panel.bridge.invoke("detect")
            }
        }
        AppButton {
            Layout.fillWidth: true; Layout.leftMargin: 8; Layout.rightMargin: 8; Layout.bottomMargin: 6
            colors: panel.colors
            text: panel.state.controlsPage === "branches" ? "Done editing peaks" : "Edit peaks…"
            enabled: panel.state.hasDataset && !panel.state.busy
            onClicked: panel.bridge.setParameter("controlsPage", panel.state.controlsPage === "branches" ? "find" : "branches")
        }
    }
}
