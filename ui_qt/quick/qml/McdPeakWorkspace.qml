pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    objectName: "quick_mcd_peak_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    color: colors.inset
    onColorsChanged: if (bridge) bridge.setPalette(colors)
    Component.onCompleted: if (bridge) bridge.setPalette(colors)
    Accessible.name: "MCD peak energies, derivatives and valley splitting"
    component Action: AppButton {
        required property string command
        colors: page.colors
        compact: true
        enabled: !page.state.busy
        onClicked: page.bridge.invoke(command)
    }
    ColumnLayout {
        anchors.fill: parent
        spacing: 4
        Flow {
            Layout.fillWidth: true
            Layout.margins: 6
            spacing: 4
            Action { text: "‹ Feature"; command: "previous-peak" }
            AppParameterField {
                width: Math.max(140, Math.min(300, page.width - 24))
                bridge: page.bridge
                colors: page.colors
                showLabel: false
                descriptor: ({type: "choice", key: "selectedPeak", value: page.state.selectedPeak,
                    options: (page.state.peakCandidates || []).map(function(row) { return {value: row.id, label: row.label} }), enabled: !page.state.busy})
            }
            Action { text: "Feature ›"; command: "next-peak" }
            Action { text: "‹ B"; command: "previous-field" }
            Action { text: "B ›"; command: "next-field" }
            Action { text: "Analyze"; command: "analyze-peaks" }
            Action { text: "Stop"; command: "cancel"; visible: page.state.canCancel; enabled: true }
        }
        Flow {
            Layout.fillWidth: true
            Layout.leftMargin: 6
            spacing: 4
            visible: page.state.compactInspection && page.state.inspectionVisible
            Repeater {
                model: [{key: "inspection", label: "Spectra + derivative"}, {key: "result", label: "Result versus B"}, {key: "maps", label: "Channel maps"}]
                delegate: AppButton {
                    required property var modelData
                    colors: page.colors
                    compact: true
                    text: modelData.label
                    primary: page.state.values?.peakPanel === modelData.key
                    onClicked: page.bridge.setParameter("peakPanel", modelData.key)
                }
            }
        }
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 160
            PlotSurface { objectName: "mcd_peak_plot_surface"; anchors.fill: parent; backend: page.bridge ? page.bridge.plotBackend : null; visible: page.state.inspectionVisible }
            AppInspectionDivider {
                anchors.fill: parent
                visible: page.state.inspectionVisible && page.state.compactInspection && (page.state.values?.peakPanel === "maps" || page.state.values?.showPeakMaps) && page.state.values?.peakPanel !== "result"
                bridge: page.bridge
                state: page.state
                colors: page.colors
            }
            Text {
                anchors.centerIn: parent
                width: Math.max(100, Math.min(520, parent.width - 36))
                visible: !page.state.inspectionVisible
                text: page.state.workspaceMessage || "Choose an MCD sweep."
                color: page.state.error ? page.colors.error : page.colors.muted
                wrapMode: Text.Wrap
                horizontalAlignment: Text.AlignHCenter
            }
        }
        Text { Layout.fillWidth: true; Layout.margins: 6; text: page.state.error || page.state.status || ""; color: page.state.error ? page.colors.error : page.colors.muted; wrapMode: Text.Wrap; font.pointSize: 9; maximumLineCount: 3; elide: Text.ElideRight }
    }
    McdReview {
        bridge: page.bridge
        colors: page.colors
        hostWidth: page.width
        hostHeight: page.height
        visible: !!page.state.sheet && page.state.sheet !== "peak-table" && page.state.sheet !== "valley-table"
    }
    Basic.Popup {
        id: table
        objectName: "mcd_peak_table_sheet"
        width: Math.max(280, Math.min(900, page.width - 16))
        height: Math.max(260, Math.min(760, page.height - 16))
        x: (page.width - width) / 2
        y: (page.height - height) / 2
        modal: true
        focus: true
        visible: page.state.sheet === "peak-table" || page.state.sheet === "valley-table"
        closePolicy: Basic.Popup.CloseOnEscape
        onClosed: if (page.bridge && (page.state.sheet === "peak-table" || page.state.sheet === "valley-table")) page.bridge.invoke("close-sheet")
        background: Rectangle { color: page.colors.surface; radius: 7; border.color: page.colors.border }
        ColumnLayout {
            anchors.fill: parent
            Text { text: page.state.sheet === "peak-table" ? "Measured-channel peak trajectories" : "Selected optical peak pair: derived valley quantities"; color: page.colors.text; wrapMode: Text.Wrap; Layout.fillWidth: true; font.weight: Font.DemiBold }
            Basic.ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: availableWidth
                clip: true
                Basic.TextArea { text: JSON.stringify(page.state.sheet === "peak-table" ? page.state.peakRows || [] : page.state.valleyRows || [], null, 2); readOnly: true; selectByMouse: true; color: page.colors.text; wrapMode: Text.Wrap; background: Rectangle { color: page.colors.inset } }
            }
            Action { text: "Close"; command: "close-sheet"; Layout.alignment: Qt.AlignRight }
        }
    }
}
