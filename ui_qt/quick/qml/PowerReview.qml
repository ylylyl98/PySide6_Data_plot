pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    objectName: "power_combination_review"
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    readonly property var review: state.review
    readonly property bool advanced: review.state.advanced === true
    property var reviewKeys: []
    property string rowSignature: ""
    function synchronizeRows() {
        let keys = review.details.map(function(row) { return row.key })
        let signature = JSON.stringify(keys)
        if (signature !== rowSignature) { reviewKeys = keys; rowSignature = signature }
    }
    onReviewChanged: synchronizeRows()
    Component.onCompleted: synchronizeRows()
    color: colors.inset
    Accessible.name: "Review power sweep combination"
    function setRow(key, field, value) { bridge.setParameter("review.row:" + encodeURIComponent(key) + ":" + field, value) }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 6
        Flow {
            Layout.fillWidth: true
            spacing: 5
            AppButton { objectName: "power_review_back"; colors: page.colors; text: "Back to selection"; onClicked: page.bridge.invoke("back-to-selection") }
            AppButton { colors: page.colors; text: "Return to plot"; enabled: Object.keys(page.state.activeSelection || {}).length > 0; onClicked: page.bridge.invoke("cancel-selection") }
            AppButton { objectName: "power_review_save"; colors: page.colors; text: "Save and plot"; primary: true; enabled: page.review.canSave; onClicked: page.bridge.invoke("save-combination") }
            AppButton { visible: page.review.needsExistingChoice; colors: page.colors; text: "Open existing"; onClicked: page.bridge.invoke("open-existing-combination") }
            AppButton { visible: page.review.needsExistingChoice; colors: page.colors; text: "Create another"; onClicked: page.bridge.invoke("create-another-combination") }
        }
        Text { Layout.fillWidth: true; text: "Every additional sweep is corrected directly against the reference. Rejected estimates remain visible and cannot be saved."; color: page.colors.muted; wrapMode: Text.Wrap }
        Basic.ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(260, Math.max(100, page.height * .36))
            contentWidth: availableWidth
            ColumnLayout {
                width: parent.width
                spacing: 6
                Repeater {
                    model: page.reviewKeys
                    delegate: Rectangle {
                        id: row
                        required property var modelData
                        required property int index
                        readonly property var rowData: page.review.details.find(function(row) { return row.key === modelData }) || ({key: modelData, title: "", powerMin: 0, powerMax: 0, count: 0, flaggedRows: [], acquisition: [], mode: "none", factor: 1, background: 0, excludedText: ""})
                        Layout.fillWidth: true
                        implicitHeight: rowContent.implicitHeight + 14
                        color: page.colors.control
                        border.color: row.rowData.error ? page.colors.error : page.colors.border
                        radius: 4
                        GridLayout {
                            id: rowContent
                            anchors.fill: parent
                            anchors.margins: 7
                            columns: page.advanced && width > 700 ? 4 : 2
                            columnSpacing: 8
                            rowSpacing: 4
                            Text { Layout.columnSpan: rowContent.columns; Layout.fillWidth: true; text: "Sweep " + (row.index + 1) + " · " + row.rowData.title + "\n" + row.rowData.powerMin.toPrecision(6) + "–" + row.rowData.powerMax.toPrecision(6) + " uW · " + row.rowData.count + " rows"; color: page.colors.text; wrapMode: Text.Wrap; font.weight: Font.DemiBold }
                            ColumnLayout {
                                visible: page.advanced
                                Layout.fillWidth: true
                                Text { text: "Correction"; color: page.colors.muted }
                                AppComboBox {
                                    colors: page.colors
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    property var values: ["none", "known", "estimate"]
                                    model: ["No scaling", "Known factor", "Estimate from overlap"]
                                    currentIndex: values.indexOf(row.rowData.mode)
                                    enabled: !page.review.state.automatic && row.rowData.key !== page.review.state.reference
                                    Accessible.name: "Sweep " + (row.index + 1) + " correction"
                                    onActivated: function(index) { page.setRow(row.rowData.key, "mode", values[index]) }
                                }
                            }
                            ColumnLayout {
                                visible: page.advanced
                                Layout.fillWidth: true
                                Text { text: "Background"; color: page.colors.muted }
                                AppNumberField {
                                    Layout.fillWidth: true
                                    bridge: page.bridge; colors: page.colors
                                    descriptor: ({key: "review.row:" + encodeURIComponent(row.rowData.key) + ":background", label: "Sweep " + (row.index + 1) + " background", type: "number", value: row.rowData.background, min: -1e12, max: 1e12, step: .01, decimals: 6, enabled: page.review.state.manualBackground === true})
                                }
                            }
                            ColumnLayout {
                                Layout.fillWidth: true
                                Text { text: "Multiplier"; color: page.colors.muted }
                                AppNumberField {
                                    Layout.fillWidth: true
                                    bridge: page.bridge; colors: page.colors
                                    descriptor: ({key: "review.row:" + encodeURIComponent(row.rowData.key) + ":factor", label: "Sweep " + (row.index + 1) + " correction multiplier", type: "number", value: row.rowData.factor, min: 1e-9, max: 1e9, step: .01, decimals: 9, enabled: !page.review.state.automatic && row.rowData.mode === "known" && row.rowData.key !== page.review.state.reference})
                                }
                            }
                            ColumnLayout {
                                visible: page.advanced
                                Layout.fillWidth: true
                                Text { text: "Exclude rows (1-based)"; color: page.colors.muted; wrapMode: Text.Wrap; Layout.fillWidth: true }
                                AppParameterField {
                                    Layout.fillWidth: true
                                    bridge: page.bridge; colors: page.colors; showLabel: false
                                    descriptor: ({key: "review.row:" + encodeURIComponent(row.rowData.key) + ":excluded", label: "Sweep " + (row.index + 1) + " excluded rows", type: "text", value: row.rowData.excludedText})
                                }
                            }
                            Text {
                                Layout.columnSpan: rowContent.columns
                                Layout.fillWidth: true
                                visible: !!row.rowData.error || !!row.rowData.quality || row.rowData.flaggedRows.length > 0
                                text: (row.rowData.error || "") + (row.rowData.quality ? "\n" + (row.rowData.quality.accepted ? row.rowData.quality.provisional ? "Provisional" : "Accepted" : "REJECTED") + " · " + row.rowData.quality.pairs.length + " calibration pairs · scatter " + (row.rowData.quality.relative_scatter * 100).toFixed(2) + "% · mismatch " + (row.rowData.quality.max_relative_residual * 100).toFixed(2) + "%" : "") + (row.rowData.flaggedRows.length ? "\nStage discrepancy flags: " + row.rowData.flaggedRows.join(", ") : "")
                                color: row.rowData.error ? page.colors.error : page.colors.muted
                                wrapMode: Text.Wrap
                            }
                            Text { Layout.columnSpan: rowContent.columns; Layout.fillWidth: true; visible: page.advanced; text: row.rowData.acquisitionText || "Acquisition information not recorded"; color: page.colors.muted; wrapMode: Text.Wrap; font.pixelSize: 11 }
                        }
                    }
                }
            }
        }
        Basic.CheckBox {
            objectName: "power_review_compatible"
            Layout.fillWidth: true
            text: "I reviewed compatibility, corrections, backgrounds and flagged measurements."
            checked: page.review.compatible
            Accessible.name: text
            onToggled: page.bridge.setParameter("review.compatible", checked)
            contentItem: Text { text: parent.text; leftPadding: parent.indicator.width + parent.spacing; color: page.colors.text; wrapMode: Text.Wrap; verticalAlignment: Text.AlignVCenter; font: parent.font }
        }
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 130
            PlotSurface { objectName: "power_review_plot"; anchors.fill: parent; backend: page.bridge.plotBackend }
            Text { anchors.centerIn: parent; visible: !page.review.ready; text: page.state.busy ? "Computing review…" : page.state.status; color: page.colors.muted; wrapMode: Text.Wrap; width: Math.max(100, parent.width - 40); horizontalAlignment: Text.AlignHCenter }
        }
        Text { Layout.fillWidth: true; text: page.state.status; color: page.state.error ? page.colors.error : page.colors.muted; wrapMode: Text.Wrap }
    }
}
