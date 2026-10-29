pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Basic.Popup {
    id: dialog
    objectName: "mcd_review_sheet"
    required property var bridge
    required property var colors
    property real hostWidth: 900
    property real hostHeight: 700
    readonly property var state: bridge ? bridge.snapshot : ({})
    readonly property var values: state.values || ({})
    property real manualCenter: values.center || 0
    property real protectionLow: (values.center || 0) - .01
    property real protectionHigh: (values.center || 0) + .01
    property string mcdLink: ""
    property string spectrumLink: ""
    readonly property var mcdCandidates: (state.allCandidates || []).filter(function(row) { return row.domain === "mcd" && row.kind !== "window" })
    readonly property var spectrumCandidates: (state.allCandidates || []).filter(function(row) { return row.domain === "spectrum" })
    width: Math.max(280, Math.min(900, hostWidth - 16))
    height: Math.max(260, Math.min(760, hostHeight - 16))
    x: (hostWidth - width) / 2
    y: (hostHeight - height) / 2
    modal: true
    focus: true
    visible: !!state.sheet
    padding: 12
    closePolicy: Basic.Popup.CloseOnEscape
    onClosed: if (bridge && state.sheet) bridge.invoke("close-sheet")
    background: Rectangle { color: dialog.colors.surface; radius: 7; border.color: dialog.colors.border }

    component Action: AppButton {
        required property string command
        colors: dialog.colors
        compact: true
        onClicked: dialog.bridge.invoke(command)
    }
    component Note: Text { Layout.fillWidth: true; color: dialog.colors.muted; wrapMode: Text.Wrap; font.pointSize: 9 }
    component NumberInput: AppNumberField { colors: dialog.colors; implicitWidth: 110 }
    ColumnLayout {
        anchors.fill: parent
        spacing: 7
        Text { Layout.fillWidth: true; text: dialog.state.sheet === "protection" ? "Select MCD feature-protection regions" : dialog.state.sheet === "retained" ? "Retained numerical results" : dialog.state.sheet === "manual" ? "Manual feature center" : dialog.state.sheet === "connections" ? "Manual MCD ↔ spectral associations" : dialog.state.sheet === "slope-details" ? "Branch slope details" : "MCD pair diagnostics"; color: dialog.colors.text; font.weight: Font.DemiBold; font.pointSize: 11 }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: dialog.state.sheet === "protection"
            Note { text: "Drag across each resonance to protect it. The active MCD(B) window is also excluded. Every sufficiently wide unprotected interval becomes a background-fit band." }
            PlotSurface { objectName: "mcd_protection_surface"; Layout.fillWidth: true; Layout.fillHeight: true; Layout.minimumHeight: 130; backend: dialog.bridge ? dialog.bridge.reviewBackend : null }
            Basic.ScrollView {
                id: regionsScroll
                Layout.fillWidth: true
                Layout.preferredHeight: Math.min(160, (dialog.state.reviewRegions || []).length * 42 + 4)
                contentWidth: availableWidth
                clip: true
                ColumnLayout {
                    width: regionsScroll.availableWidth
                    Repeater {
                        model: (dialog.state.reviewRegions || []).length
                        delegate: RowLayout {
                            id: region
                            required property int index
                            readonly property var record: dialog.state.reviewRegions[index] || ({low: 0, high: 1, enabled: true})
                            Layout.fillWidth: true
                            QtObject {
                                id: regionBridge
                                function setParameter(key, value) { dialog.bridge.editProtection(region.index, key === "low" ? value : region.record.low, key === "high" ? value : region.record.high, region.record.enabled) }
                            }
                            Basic.CheckBox { checked: region.record.enabled; text: String(region.index + 1); onToggled: dialog.bridge.editProtection(region.index, region.record.low, region.record.high, checked) }
                            NumberInput { Layout.fillWidth: true; descriptor: ({key: "low", value: region.record.low, decimals: 8, displayDecimals: 4, step: .001}); bridge: regionBridge }
                            Text { text: "to"; color: dialog.colors.muted }
                            NumberInput { Layout.fillWidth: true; descriptor: ({key: "high", value: region.record.high, decimals: 8, displayDecimals: 4, step: .001}); bridge: regionBridge }
                            AppButton { colors: dialog.colors; compact: true; text: "Remove"; onClicked: dialog.bridge.removeProtection(region.index) }
                        }
                    }
                }
            }
            RowLayout {
                QtObject { id: addBridge; function setParameter(key, value) { if (key === "low") dialog.protectionLow = value; else dialog.protectionHigh = value } }
                NumberInput { Layout.fillWidth: true; descriptor: ({key: "low", value: dialog.protectionLow, decimals: 8, displayDecimals: 4, step: .001}); bridge: addBridge }
                Text { text: "to"; color: dialog.colors.muted }
                NumberInput { Layout.fillWidth: true; descriptor: ({key: "high", value: dialog.protectionHigh, decimals: 8, displayDecimals: 4, step: .001}); bridge: addBridge }
                AppButton { colors: dialog.colors; compact: true; text: "Add region"; onClicked: dialog.bridge.addProtection(dialog.protectionLow, dialog.protectionHigh) }
            }
            Note { text: dialog.state.reviewSummary || ""; maximumLineCount: 5; elide: Text.ElideRight }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: dialog.state.sheet === "manual"
            Note { text: "The center seeds tracking in the raw + channel. It does not establish a K / K′ assignment or a valid spectral association." }
            QtObject { id: manualBridge; function setParameter(key, value) { dialog.manualCenter = value } }
            NumberInput { Layout.fillWidth: true; descriptor: ({key: "center", value: dialog.manualCenter, decimals: 8, step: .001}); bridge: manualBridge }
            AppButton { colors: dialog.colors; text: "Add manual candidate"; primary: true; onClicked: dialog.bridge.addManualCandidate(dialog.manualCenter) }
            Item { Layout.fillHeight: true }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: dialog.state.sheet === "connections"
            Note { text: "Choose an MCD feature and a spectrum feature explicitly. Associations preserve their manual provenance; unresolved or ambiguous relationships do not create a splitting result." }
            QtObject {
                id: associationBridge
                function setParameter(key, value) { if (key === "mcdLink") dialog.mcdLink = value; else dialog.spectrumLink = value }
            }
            AppParameterField {
                Layout.fillWidth: true
                descriptor: ({type: "choice", key: "mcdLink", label: "MCD feature", value: dialog.mcdLink || dialog.mcdCandidates[0]?.id || "",
                    options: dialog.mcdCandidates.map(function(row) { return {value: row.id, label: row.id + " · " + Number(row.energy_ev).toPrecision(7) + " eV"} })})
                bridge: associationBridge
                colors: dialog.colors
            }
            AppParameterField {
                Layout.fillWidth: true
                descriptor: ({type: "choice", key: "spectrumLink", label: "Spectrum feature", value: dialog.spectrumLink || dialog.spectrumCandidates[0]?.id || "",
                    options: dialog.spectrumCandidates.map(function(row) { return {value: row.id, label: row.source + " · " + Number(row.energy_ev).toPrecision(7) + " eV · " + (row.branch || "")} })})
                bridge: associationBridge
                colors: dialog.colors
            }
            RowLayout {
                AppButton { colors: dialog.colors; text: "Connect"; onClicked: dialog.bridge.setManualLink(dialog.mcdLink || dialog.mcdCandidates[0]?.id || "", dialog.spectrumLink || dialog.spectrumCandidates[0]?.id || "", true) }
                AppButton { colors: dialog.colors; text: "Remove connection"; onClicked: dialog.bridge.setManualLink(dialog.mcdLink || dialog.mcdCandidates[0]?.id || "", dialog.spectrumLink || dialog.spectrumCandidates[0]?.id || "", false) }
            }
            Basic.ScrollView { Layout.fillWidth: true; Layout.fillHeight: true; clip: true; contentWidth: availableWidth; Basic.TextArea { text: JSON.stringify(dialog.state.manualLinks || [], null, 2); color: dialog.colors.text; readOnly: true; wrapMode: Text.Wrap } }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: dialog.state.sheet === "retained"
            Note { text: "Retained results are independent numerical snapshots. Check exactly the items to export; stale or uncomputed items cannot be exported." }
            Basic.ScrollView {
                id: retainedScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: availableWidth
                clip: true
                ColumnLayout {
                    width: retainedScroll.availableWidth
                    Repeater {
                        model: dialog.state.retained || []
                        delegate: RowLayout {
                            required property var modelData
                            Layout.fillWidth: true
                            Basic.CheckBox { checked: parent.modelData.included; onToggled: dialog.bridge.includeRetained(parent.modelData.kind, parent.modelData.id, checked) }
                            AppButton {
                                colors: dialog.colors
                                Layout.fillWidth: true
                                text: parent.modelData.kind + " · " + parent.modelData.label + " · " + parent.modelData.status
                                checked: dialog.state.selectedRetainedId === parent.modelData.id && dialog.state.selectedRetainedKind === parent.modelData.kind
                                onClicked: dialog.bridge.selectRetained(parent.modelData.kind, parent.modelData.id)
                            }
                        }
                    }
                }
            }
            Flow { Layout.fillWidth: true; spacing: 5; Action { command: "inspect-retained"; text: "Inspect selected" } Action { command: "update-retained"; text: "Update selected" } Action { command: "remove-retained"; text: "Remove selected" } Action { command: "export-retained"; text: "Export checked results"; primary: true } }
        }
        Basic.ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: dialog.state.sheet === "diagnostics" || dialog.state.sheet === "slope-details"
            clip: true
            contentWidth: availableWidth
            Basic.TextArea { text: JSON.stringify(dialog.state.sheet === "slope-details" ? dialog.state.slopes || {} : dialog.state.diagnostics || [], null, 2); readOnly: true; selectByMouse: true; color: dialog.colors.text; wrapMode: Text.Wrap; font.pointSize: 9; background: Rectangle { color: dialog.colors.inset } }
        }
        Note { visible: !!dialog.state.error; text: dialog.state.error || ""; color: dialog.colors.error }
        RowLayout {
            Layout.fillWidth: true
            Item { Layout.fillWidth: true }
            Action { command: "close-sheet"; text: "Close" }
            Action { command: "protection-apply"; text: "Use selected regions / recalculate"; primary: true; visible: dialog.state.sheet === "protection"; enabled: !!dialog.state.reviewSummary && dialog.state.reviewSummary !== "Protection review is calculating." }
        }
    }
}
