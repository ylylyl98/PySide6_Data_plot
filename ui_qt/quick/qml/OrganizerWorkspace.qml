pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    objectName: "quick_organizer_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    color: colors.inset
    component Action: AppButton {
        required property string command
        colors: page.colors
        compact: true
        enabled: page.state.canClose
        onClicked: page.bridge.invoke(command)
    }
    ColumnLayout {
        anchors.fill: parent
        spacing: 4
        Flow {
            Layout.fillWidth: true
            Layout.margins: 5
            spacing: 4
            Repeater {
                model: [{key: "mcd", label: "MCD versus B"}, {key: "slopes", label: "Window energy / slopes"}, {key: "cw", label: "CW fit"}, {key: "field", label: "Field fits"}, {key: "scan", label: "Range scan"}, {key: "methods", label: "CW methods"}, {key: "theta", label: "θCW comparison"}]
                delegate: AppButton {
                    required property var modelData
                    colors: page.colors
                    compact: true
                    text: modelData.label
                    primary: page.state.values.view === modelData.key
                    onClicked: page.bridge.setParameter("view", modelData.key)
                }
            }
            Action { text: "Stop"; command: "cancel"; visible: page.state.canCancel }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 4
            Item {
                Layout.fillWidth: true
                Layout.fillHeight: true
                PlotSurface { objectName: "organizer_plot_surface"; anchors.fill: parent; backend: page.bridge.plotBackend; visible: !page.state.previewPending }
                Rectangle {
                    anchors.fill: parent
                    visible: !!page.state.previewPending
                    color: page.colors.inset
                    Text { anchors.centerIn: parent; text: "Loading selected MCD series…"; color: page.colors.muted }
                }
            }
            Basic.ScrollView {
                id: conditionScroll
                Layout.preferredWidth: Math.min(300, page.width * .37)
                Layout.minimumWidth: 180
                Layout.fillHeight: true
                visible: page.state.values.showConditions
                contentWidth: availableWidth
                clip: true
                ColumnLayout {
                    width: conditionScroll.availableWidth
                    spacing: 5
                    Text { text: "Conditions · Ctrl / Shift: select several"; color: page.colors.text; wrapMode: Text.Wrap; Layout.fillWidth: true; font.weight: Font.DemiBold }
                    Flow { Layout.fillWidth: true; spacing: 4; Action { text: "Exclude"; command: "exclude-selected" } Action { text: "Restore"; command: "restore-selected" } Action { text: "Restore all"; command: "restore-all" } Action { text: "Assign group"; command: "assign-selected" } }
                    Repeater {
                        model: page.state.conditions || []
                        delegate: Rectangle {
                            id: condition
                            required property var modelData
                            Layout.fillWidth: true
                            implicitHeight: content.implicitHeight + 10
                            color: modelData.selected ? page.colors.selected : modelData.included ? page.colors.control : page.colors.surface
                            radius: 4
                            border.color: modelData.focused ? page.colors.accent : page.colors.border
                            RowLayout {
                                id: content
                                anchors.fill: parent
                                anchors.margins: 5
                                Basic.CheckBox { checked: condition.modelData.included; onToggled: page.bridge.includeCondition(condition.modelData.id, checked); Accessible.name: "Include " + condition.modelData.label }
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    Text { Layout.fillWidth: true; text: condition.modelData.label; color: condition.modelData.color; wrapMode: Text.Wrap; font.pointSize: 9 }
                                    Text { Layout.fillWidth: true; text: condition.modelData.slopes; color: page.colors.muted; wrapMode: Text.Wrap; font.pointSize: 8 }
                                    AppButton { Layout.fillWidth: true; colors: page.colors; compact: true; text: condition.modelData.focused ? "Focused" : "Select / focus"; onClicked: page.bridge.selectCondition(condition.modelData.id, false) }
                                }
                            }
                        }
                    }
                }
            }
        }
        Text { Layout.fillWidth: true; Layout.margins: 5; text: page.state.hover || (page.state.values.view === "cw" ? page.state.cwSummary : page.state.status); color: page.state.error ? page.colors.error : page.colors.muted; wrapMode: Text.Wrap; font.pointSize: 9; maximumLineCount: 4; elide: Text.ElideRight }
    }
    Basic.Popup {
        id: sheet
        objectName: "organizer_review_sheet"
        width: Math.max(280, Math.min(1000, page.width - 16))
        height: Math.max(260, Math.min(820, page.height - 16))
        x: (page.width - width) / 2
        y: (page.height - height) / 2
        modal: true
        focus: true
        visible: !!page.state.sheet
        closePolicy: Basic.Popup.CloseOnEscape
        onClosed: if (page.bridge && page.state.sheet) page.bridge.invoke("close-sheet")
        padding: 10
        background: Rectangle { color: page.colors.surface; radius: 7; border.color: page.colors.border }
        ColumnLayout {
            anchors.fill: parent
            Text { Layout.fillWidth: true; text: page.state.sheet === "theta" ? "Explicit optical group mapping and θCW comparison" : page.state.sheet === "edit-groups" ? "Saved integration energy group assignments" : page.state.sheet === "cw-details" ? "CW fits and diagnostics" : "Saved numerical results and diagnostics"; color: page.colors.text; font.weight: Font.DemiBold; wrapMode: Text.Wrap }
            Basic.ScrollView {
                id: groupScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                visible: page.state.sheet === "theta" || page.state.sheet === "edit-groups"
                contentWidth: availableWidth
                clip: true
                ColumnLayout {
                    width: groupScroll.availableWidth
                    Repeater {
                        model: page.state.sheet === "theta" ? page.state.thetaMapping || [] : page.state.conditions || []
                        delegate: ColumnLayout {
                            id: mapping
                            required property var modelData
                            Layout.fillWidth: true
                            Text { Layout.fillWidth: true; text: mapping.modelData.label; color: page.colors.text; wrapMode: Text.Wrap; font.pointSize: 9 }
                            QtObject { id: rowBridge; function setParameter(key, value) { if (page.state.sheet === "theta") page.bridge.setThetaGroup(mapping.modelData.id, value); else page.bridge.assignRecordGroup(mapping.modelData.id, value) } }
                            AppParameterField {
                                Layout.fillWidth: true
                                colors: page.colors
                                bridge: rowBridge
                                descriptor: page.state.sheet === "theta" ? ({type: "choice", key: "group", label: "Matching optical group", value: mapping.modelData.selected, options: (mapping.modelData.groups || []).map(function(name) { return {value: name, label: name} })}) : page.state.values.fixedGroupCount ? ({type: "choice", key: "group", label: "Assigned energy group", value: mapping.modelData.group.indexOf("Unassigned ") === 0 ? "" : mapping.modelData.group, options: [{value: "", label: "Unassigned"}].concat(Array.from({length: page.state.values.groupCount}, function(_, index) { let name = "Group " + (index + 1); return {value: name, label: name} }))}) : ({type: "text", key: "group", label: "Assigned energy group (blank = unassigned)", value: mapping.modelData.group.indexOf("Unassigned ") === 0 ? "" : mapping.modelData.group})
                            }
                        }
                    }
                    Flow {
                        Layout.fillWidth: true
                        visible: page.state.sheet === "theta"
                        spacing: 4
                        Repeater {
                            model: page.state.thetaAvailableGroups || []
                            delegate: Basic.CheckBox { required property string modelData; text: modelData; checked: (page.state.thetaGroups || []).indexOf(modelData) >= 0; onToggled: page.bridge.setThetaVisible(modelData, checked) }
                        }
                    }
                }
            }
            Basic.ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredHeight: page.state.sheet === "theta" ? 180 : 500
                contentWidth: availableWidth
                clip: true
                visible: page.state.sheet !== "edit-groups"
                Basic.TextArea {
                    readOnly: true
                    selectByMouse: true
                    color: page.colors.text
                    wrapMode: Text.Wrap
                    text: page.state.sheet === "cw-details" ? page.state.cwSummary : page.state.sheet === "methods" ? JSON.stringify(page.state.methodRows || [], null, 2) : page.state.sheet === "scan" ? JSON.stringify(page.state.scanRows || [], null, 2) : page.state.sheet === "theta" ? JSON.stringify(page.state.thetaRows || [], null, 2) + "\n" + page.state.thetaDiagnostics : JSON.stringify(page.state.details || [], null, 2)
                    background: Rectangle { color: page.colors.inset }
                }
            }
            Text { Layout.fillWidth: true; text: page.state.error || page.state.saveError || ""; color: page.colors.error; wrapMode: Text.Wrap }
            RowLayout {
                Layout.fillWidth: true
                Action { text: "Run θCW comparison"; command: "compare-theta"; visible: page.state.sheet === "theta" }
                Item { Layout.fillWidth: true }
                Action { text: "Close"; command: "close-sheet" }
            }
        }
    }
}
