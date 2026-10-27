pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    color: colors.inset
    objectName: "quick_mcd_workspace"
    onColorsChanged: if (bridge) bridge.setPalette(colors)
    Component.onCompleted: if (bridge) bridge.setPalette(colors)

    component Action: AppButton {
        required property string command
        colors: page.colors
        compact: true
        enabled: !page.state.busy
        onClicked: page.bridge.invoke(command)
    }
    ColumnLayout {
        anchors.fill: parent
        spacing: 5
        Flow {
            Layout.fillWidth: true
            Layout.margins: 6
            spacing: 5
            Action { text: "Recalculate"; command: "recalculate" }
            Action { text: "‹"; description: "Previous feature or window"; command: "previous-candidate"; enabled: page.state.inspectionVisible }
            AppParameterField {
                width: Math.min(260, page.width - 24)
                descriptor: ({type: "choice", key: "candidate", label: "Feature / window", value: page.state.selectedCandidate,
                    options: (page.state.candidates || []).map(function(row) { return {value: row.id, label: row.label} }), enabled: !page.state.busy})
                bridge: page.bridge
                colors: page.colors
                showLabel: false
            }
            Action { text: "›"; description: "Next feature or window"; command: "next-candidate"; enabled: page.state.inspectionVisible }
            Action { text: "+ Manual"; command: "manual-candidate"; enabled: page.state.inspectionVisible }
            Action { text: "Pair diagnostics"; command: "diagnostics"; enabled: page.state.inspectionVisible }
            Action { text: "Retained results"; command: "retained" }
            Action { text: "Stop"; command: "cancel"; visible: page.state.canCancel; enabled: true }
        }
        Flow {
            Layout.fillWidth: true
            Layout.leftMargin: 6
            Layout.rightMargin: 6
            spacing: 4
            visible: page.state.inspectionVisible && !page.state.compactInspection
            Repeater {
                model: ["Raw", "Corrected"]
                delegate: AppButton {
                    required property string modelData
                    colors: page.colors
                    compact: true
                    text: modelData + " spectra"
                    checked: page.state.values?.spectraSource === modelData
                    onClicked: page.bridge.setParameter("spectraSource", modelData)
                }
            }
            AppButton { colors: page.colors; compact: true; text: "Inc sweep"; checked: page.state.values?.showIncreasing; onClicked: page.bridge.setParameter("showIncreasing", !page.state.values.showIncreasing) }
            AppButton { colors: page.colors; compact: true; text: "Dec sweep"; checked: page.state.values?.showDecreasing; onClicked: page.bridge.setParameter("showDecreasing", !page.state.values.showDecreasing) }
            Repeater {
                model: [{key: "shift", label: "Shift"}, {key: "energy", label: "Energy"}, {key: "splitting", label: "Splitting"}]
                delegate: AppButton {
                    required property var modelData
                    colors: page.colors
                    compact: true
                    text: modelData.label
                    checked: page.state.values?.featureMetric === modelData.key
                    onClicked: page.bridge.setParameter("featureMetric", modelData.key)
                }
            }
        }
        Flow {
            Layout.fillWidth: true
            Layout.margins: 6
            spacing: 4
            visible: page.state.compactInspection && page.state.inspectionVisible
            Repeater {
                model: [{key: "map", label: "MCD map"}, {key: "spectra", label: "Paired spectra"}, {key: "mcd", label: "MCD spectrum"}, {key: "trace", label: "MCD(B)"}, {key: "feature", label: "Feature(B)"}]
                delegate: AppButton {
                    required property var modelData
                    colors: page.colors
                    compact: true
                    text: modelData.label
                    primary: page.state.panel === modelData.key
                    onClicked: page.bridge.setParameter("panel", modelData.key)
                }
            }
        }
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 160
            PlotSurface { objectName: "mcd_plot_surface"; anchors.fill: parent; anchors.bottomMargin: spectrumInspection.visible ? spectrumInspection.footerHeight : 0; backend: page.bridge ? page.bridge.plotBackend : null; visible: page.state.inspectionVisible }
            AppInspectionDivider {
                id: spectrumInspection
                anchors.fill: parent
                visible: page.state.inspectionVisible && page.state.compactInspection && page.state.panel === "map"
                bridge: page.bridge
                state: page.state
                colors: page.colors
            }
            Text {
                objectName: "mcd_workspace_message"
                anchors.centerIn: parent
                width: Math.max(100, Math.min(540, parent.width - 36))
                visible: !page.state.inspectionVisible
                text: page.state.workspaceMessage || "Choose an MCD B-sweep CSV."
                color: page.state.error ? page.colors.error : page.colors.muted
                wrapMode: Text.Wrap
                horizontalAlignment: Text.AlignHCenter
                font.pointSize: 11
            }
        }
        Text { Layout.fillWidth: true; Layout.margins: 6; text: page.state.windowSummary || ""; color: page.colors.muted; wrapMode: Text.Wrap; font.pointSize: 9 }
        Text { Layout.fillWidth: true; Layout.margins: 6; text: page.state.error || page.state.selectedFeatureStatus || page.state.summary || ""; color: page.state.error ? page.colors.error : page.colors.muted; wrapMode: Text.Wrap; font.pointSize: 9 }
    }
    McdReview { bridge: page.bridge; colors: page.colors; hostWidth: page.width; hostHeight: page.height }
}
