import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

ColumnLayout {
    id: root
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    readonly property bool creating: state.pnMode !== "saved"
    signal chooseFilesRequested()
    signal matchingConditionsRequested()
    property bool batchExpanded: false
    spacing: 10
    clip: true
    objectName: "drr_pn_simple_workflow"

    component Note: Text {
        Layout.fillWidth: true
        color: root.colors.muted
        wrapMode: Text.Wrap
        font.pointSize: 9
    }
    component Heading: Text {
        Layout.fillWidth: true
        color: root.colors.text
        font.pointSize: 10
        font.weight: Font.DemiBold
        wrapMode: Text.Wrap
    }
    component Action: AppButton {
        colors: root.colors
        compact: true
        enabled: !root.state.busy
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        Repeater {
            model: [{key: "create", title: "Create P+N background"},
                    {key: "saved", title: "Use saved background"}]
            delegate: AppButton {
                required property var modelData
                colors: root.colors
                text: modelData.title
                objectName: "drr_pn_mode_" + modelData.key
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                checkable: true
                autoExclusive: true
                checked: (root.state.pnMode || "create") === modelData.key
                enabled: !root.state.busy
                Accessible.role: Accessible.RadioButton
                Accessible.name: modelData.title
                onClicked: root.bridge.setPnMode(modelData.key)
            }
        }
    }
    Basic.SplitView {
        Layout.fillWidth: true
        Layout.fillHeight: true
        orientation: root.width < 640 ? Qt.Vertical : Qt.Horizontal
        Basic.ScrollView {
            id: sourceControls
            objectName: "drr_pn_simple_controls"
            Basic.SplitView.preferredWidth: 340
            Basic.SplitView.minimumWidth: 250
            Basic.SplitView.preferredHeight: root.height * .53
            Basic.SplitView.minimumHeight: 150
            contentWidth: availableWidth
            clip: true
            Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
            ColumnLayout {
                width: sourceControls.availableWidth - 12
                spacing: 10
                Heading { text: root.creating ? "1 · Choose P and N files" : "Choose a saved version" }
                Note {
                    visible: root.creating
                    text: "Choose one or more original files for each role. Multiple files are automatically assigned to measurements. " + (root.state.pnFramesSummary || "Both sources use all acquired frames.")
                }
                Repeater {
                    model: root.creating ? ["P", "N"] : []
                    delegate: Rectangle {
                        id: sourceCard
                        required property string modelData
                        readonly property var files: modelData === "P" ? (root.state.pFiles || []) : (root.state.nFiles || [])
                        Layout.fillWidth: true
                        implicitHeight: sourceContent.implicitHeight + 18
                        color: root.colors.control
                        border.color: root.colors.border
                        radius: 5
                        ColumnLayout {
                            id: sourceContent
                            anchors.fill: parent
                            anchors.margins: 9
                            spacing: 5
                            RowLayout {
                                Layout.fillWidth: true
                                Heading { text: sourceCard.modelData + (sourceCard.files.length > 1 ? " sources · " + sourceCard.files.length : " source") }
                                Action {
                                    objectName: "drr_pn_choose_" + sourceCard.modelData
                                    text: sourceCard.files.length ? "Change…" : "Choose…"
                                    Accessible.name: "Choose " + sourceCard.modelData + " source files"
                                    onClicked: root.bridge.invoke("choose-files:" + sourceCard.modelData)
                                }
                            }
                            Note {
                                text: sourceCard.files.length ? sourceCard.files.slice(0, 2).join("\n") + (sourceCard.files.length > 2 ? "\n+ " + (sourceCard.files.length - 2) + " more selected" : "") : "No source selected"
                                color: sourceCard.files.length ? root.colors.text : root.colors.muted
                            }
                        }
                    }
                }
                Note { visible: root.creating; text: root.state.pairSummary || "" }
                Heading { visible: root.creating && root.state.pnUsesMeasurements !== false && !root.state.pnBatchActive; text: root.state.pnSingleMeasurement ? "Measurement" : "Measurement repeats" }
                Note {
                    objectName: "drr_pn_training_summary"
                    visible: root.creating
                    text: root.state.pnTrainingSummary || "Preparing the current group's training measurements…"
                }
                Repeater {
                    model: root.creating && root.state.pnUsesMeasurements !== false && !root.state.pnBatchActive ? (root.state.pnTrainingFiles || []) : []
                    delegate: Note {
                        required property string modelData
                        text: modelData
                        color: root.colors.text
                    }
                }
                Note { visible: root.creating; text: root.state.pnMethodLabel || "P/N reconstruction · All acquired frames" }
                Basic.CheckBox {
                    objectName: "drr_pn_batch_acknowledge"
                    Layout.fillWidth: true
                    visible: root.creating && root.state.pnBatchActive && (root.state.pnBatchWarnings || []).length > 0
                    checked: root.state.pnBatchAcknowledged === true
                    enabled: !root.state.busy
                    text: "I reviewed the assignment warnings"
                    onClicked: root.bridge.acknowledgePnBatchWarnings(checked)
                    contentItem: Text {
                        text: parent.text
                        leftPadding: parent.indicator.width + parent.spacing
                        color: root.colors.text
                        wrapMode: Text.Wrap
                        font.pointSize: 9
                    }
                }
                Action {
                    objectName: "drr_pn_generate"
                    Layout.fillWidth: true
                    visible: root.creating
                    text: root.state.pnBatchActive ? (root.state.pnPreviewReady ? "Regenerate required backgrounds" : "2 · Generate required backgrounds") : root.state.pnPreviewReady ? "Generate updated background" : "2 · Generate background"
                    primary: true
                    enabled: !root.state.busy && root.state.pnCreateReady === true
                    onClicked: root.bridge.invoke("pn-generate")
                }
                Note {
                    visible: root.creating
                    text: "Generation saves a version in history and opens its preview. Apply to current group updates the main DRR result."
                }
                ColumnLayout {
                    visible: !root.creating
                    Layout.fillWidth: true
                    spacing: 8
                    Basic.ProgressBar {
                        Layout.fillWidth: true
                        indeterminate: true
                        visible: root.state.pnHistoryLoading === true
                    }
                    AppComboBox {
                        colors: root.colors
                        objectName: "drr_pn_saved_version"
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        model: root.state.choices?.version?.items || []
                        currentIndex: root.state.choices?.version?.current ?? -1
                        enabled: !root.state.busy && model.length > 0
                        Accessible.name: "Saved P+N background version"
                        onActivated: index => root.bridge.setChoice("version", index)
                    }
                    Note {
                        text: root.state.pnVersionLabel || (root.state.pnHistoryLoading ? "Loading saved backgrounds…" : "Choose a version to preview. No version is applied automatically.")
                    }
                    Note {
                        visible: root.state.pnHistoryLoaded === true && !(root.state.choices?.version?.items || []).length
                        text: "No saved P+N versions found. Choose Create P+N background to make one, or select a saved CSV below."
                    }
                    Note { text: root.state.versionStatus || ""; visible: !!root.state.pnVersionLabel }
                    Basic.CheckBox {
                        objectName: "drr_pn_use_snapshot"
                        Layout.fillWidth: true
                        visible: root.state.pnVersionNeedsSnapshot === true
                        checked: root.state.flags?.allow_stale === true
                        enabled: !root.state.busy
                        onClicked: root.bridge.setFlag("allow_stale", checked)
                        contentItem: Text {
                            text: "Use this version's saved snapshot explicitly"
                            leftPadding: parent.indicator.width + parent.spacing
                            wrapMode: Text.Wrap
                            color: root.colors.text
                            font.pointSize: 9
                        }
                    }
                    Note {
                        visible: (root.state.pFiles || []).length > 0 || (root.state.nFiles || []).length > 0
                        text: "P: " + (root.state.pFiles || []).join(", ") + "\nN: " + (root.state.nFiles || []).join(", ")
                    }
                    Action {
                        objectName: "drr_pn_refresh_history"
                        text: "Refresh saved versions"
                        enabled: !root.state.busy && !root.state.pnHistoryLoading
                        onClicked: root.bridge.invoke("pn-refresh-history")
                    }
                    Action {
                        objectName: "drr_pn_choose_saved_files"
                        Layout.fillWidth: true
                        text: "Choose saved CSV files / match measurements…"
                        onClicked: root.chooseFilesRequested()
                    }
                }
                Action {
                    objectName: "drr_pn_advanced"
                    text: "Advanced generation settings…"
                    onClicked: root.bridge.invoke("page:pn")
                }
            }
        }
        Rectangle {
            id: previewPane
            Basic.SplitView.fillWidth: true
            Basic.SplitView.fillHeight: true
            Basic.SplitView.minimumWidth: 220
            Basic.SplitView.minimumHeight: 120
            color: root.colors.control
            border.color: root.colors.border
            radius: 5
            Basic.ScrollView {
                id: previewScroll
                anchors.fill: parent
                anchors.margins: 9
                contentWidth: availableWidth
                clip: true
                Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
                ColumnLayout {
                width: previewScroll.availableWidth
                height: Math.max(previewScroll.availableHeight, implicitHeight)
                spacing: 6
                RowLayout {
                    Layout.fillWidth: true
                    visible: root.creating && root.state.pnBatchActive
                    Heading {
                        objectName: "drr_pn_batch_summary"
                        text: root.state.pnBatchSummary || "Choose P and N candidates"
                    }
                    Action {
                        objectName: "drr_pn_batch_expand"
                        text: root.batchExpanded || root.state.pnBatchUnresolved > 0 ? "Hide details" : "Review assignments"
                        onClicked: root.batchExpanded = !root.batchExpanded
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    visible: root.creating && root.state.pnBatchActive
                    Note { text: root.state.matchingConditionsSummary || "Strict conditions" }
                    Action { objectName: "drr_pn_batch_refresh"; text: "Refresh"; onClicked: root.bridge.refreshPnBatchAssignments() }
                    Action { text: "Matching conditions…"; onClicked: root.matchingConditionsRequested() }
                }
                Basic.ScrollView {
                    id: assignmentScroll
                    objectName: "drr_pn_batch_rows"
                    visible: root.creating && root.state.pnBatchActive && (root.batchExpanded || root.state.pnBatchUnresolved > 0)
                    Layout.fillWidth: true
                    Layout.preferredHeight: Math.max(80, Math.min(240, previewPane.height * .4))
                    Layout.minimumHeight: 60
                    contentWidth: availableWidth
                    clip: true
                    Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
                    ColumnLayout {
                        width: assignmentScroll.availableWidth - 10
                        spacing: 9
                        Note { visible: root.state.pnBatchLoading; text: "Finding compatible P/N pairs for the selected measurements…" }
                        Note { visible: !!root.state.pnBatchError; text: root.state.pnBatchError || ""; color: root.colors.error }
                        Repeater {
                            model: root.state.pnBatchRows || []
                            delegate: ColumnLayout {
                                id: assignmentRow
                                required property var modelData
                                required property int index
                                Layout.fillWidth: true
                                spacing: 3
                                Heading { text: assignmentRow.modelData.measurement }
                                AppComboBox {
                                    objectName: "drr_pn_batch_pair_" + assignmentRow.index
                                    colors: root.colors
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    model: assignmentRow.modelData.options
                                    currentIndex: assignmentRow.modelData.current
                                    enabled: !root.state.busy && !root.state.pnBatchLoading
                                    Accessible.name: "P/N pair for " + assignmentRow.modelData.measurement
                                    onActivated: index => root.bridge.setPnBatchPair(assignmentRow.modelData.measurement, assignmentRow.modelData.optionIndices[index])
                                }
                                Action {
                                    visible: assignmentRow.modelData.moreOptions
                                    text: "Show all " + assignmentRow.modelData.totalOptions + " compatible pairs"
                                    onClicked: root.bridge.showAllPnBatchPairs(assignmentRow.modelData.measurement)
                                }
                                Note { visible: assignmentRow.modelData.current < 0; text: "Needs a compatible P/N choice"; color: root.colors.error }
                                Note { text: assignmentRow.modelData.time }
                                Note { text: assignmentRow.modelData.evidence }
                                Note { text: assignmentRow.modelData.training || "" }
                                Note { visible: assignmentRow.modelData.warnings.length > 0; text: assignmentRow.modelData.warnings.join("\n") }
                                Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: root.colors.border }
                            }
                        }
                        Note { text: (root.state.pnBatchWarnings || []).join("\n"); visible: (root.state.pnBatchWarnings || []).length > 0 }
                    }
                }
                Heading { text: "Background preview" }
                AppComboBox {
                    objectName: "drr_pn_batch_preview"
                    colors: root.colors
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    visible: root.state.pnBatchActive && (root.state.pnBatchPreviewItems || []).length > 0
                    model: root.state.pnBatchPreviewItems || []
                    currentIndex: root.state.pnBatchPreviewIndex ?? 0
                    enabled: !root.state.busy
                    Accessible.name: "Generated background preview"
                    onActivated: index => root.bridge.setPnBatchPreview(index)
                }
                Note {
                    visible: root.state.pnPreviewReady === true
                    text: "Saved version preview · Apply explicitly to update the current group"
                }
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 120
                    PlotSurface {
                        objectName: "drr_pn_simple_plot"
                        anchors.fill: parent
                        opacity: root.state.pnPreviewReady ? 1 : 0
                        backend: root.bridge ? root.bridge.plotBackend : null
                    }
                    Text {
                        anchors.centerIn: parent
                        width: Math.max(80, parent.width - 32)
                        visible: root.state.pnPreviewReady !== true
                        text: root.creating ? "Choose P and N, then generate a background to inspect it here." : "Select a saved version to preview it here."
                        wrapMode: Text.Wrap
                        horizontalAlignment: Text.AlignHCenter
                        color: root.colors.muted
                        font.pointSize: 10
                    }
                }
                }
            }
        }
    }
}
