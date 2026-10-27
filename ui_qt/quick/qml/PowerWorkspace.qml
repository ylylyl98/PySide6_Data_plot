pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    objectName: "quick_power_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    color: colors.inset
    Accessible.name: "Power measurement and analysis workspace"
    onColorsChanged: bridge.setPalette(colors)
    Component.onCompleted: bridge.setPalette(colors)
    Loader {
        anchors.fill: parent
        sourceComponent: page.state.stage === "selection" ? selection
                       : page.state.stage === "review" ? review
                       : page.state.stage === "refit" ? refit
                       : page.state.stage === "law" ? law : plot
    }
    Component { id: selection; PowerSelection { bridge: page.bridge; colors: page.colors } }
    Component { id: review; PowerReview { bridge: page.bridge; colors: page.colors } }
    Component { id: refit; PowerAnalysisRefit { bridge: page.bridge; colors: page.colors } }
    Component { id: law; PowerAnalysisLaw { bridge: page.bridge; colors: page.colors } }
    Component {
        id: plot
        ColumnLayout {
            spacing: 0
            Flow {
                Layout.fillWidth: true
                Layout.margins: 8
                spacing: 5
                AppButton { colors: page.colors; text: "Measurements…"; onClicked: page.bridge.invoke("choose-group") }
                AppButton { objectName: "power_view_intensity"; colors: page.colors; text: "Intensity"; primary: page.state.mode !== "vp"; onClicked: page.bridge.setParameter("mode", page.state.vpAvailable ? "intensity" : "single") }
                AppButton { objectName: "power_view_vp"; colors: page.colors; text: "VP"; primary: page.state.mode === "vp"; enabled: page.state.vpAvailable; onClicked: page.bridge.setParameter("mode", "vp") }
                AppButton { colors: page.colors; text: "Fit power law…"; enabled: page.state.analysisEnabled && page.state.analysisExportReady; onClicked: page.bridge.invoke("analysis.law") }
            }
            Item {
                id: plotArea
                Layout.fillWidth: true
                Layout.fillHeight: true
                PlotSurface { objectName: "power_plot_surface"; anchors.fill: parent; anchors.bottomMargin: spectrumInspection.visible ? spectrumInspection.footerHeight : 0; backend: page.bridge.plotBackend; visible: page.state.inspectionVisible }
                AppInspectionDivider {
                    id: spectrumInspection
                    width: plotArea.width * (page.state.dividerRight || .96)
                    height: plotArea.height
                    visible: page.state.inspectionVisible
                    bridge: page.bridge; colors: page.colors
                    state: Object.assign({}, page.state, {dividerRight: 1, dividerLeft: (page.state.dividerLeft || .12) / Math.max(.1, page.state.dividerRight || .96)})
                }
                Text { anchors.centerIn: parent; visible: !page.state.inspectionVisible; text: page.state.busy ? "Loading Power data…" : page.state.workspaceMessage; color: page.state.error ? page.colors.error : page.colors.muted; width: Math.max(100, parent.width - 48); wrapMode: Text.Wrap; horizontalAlignment: Text.AlignHCenter }
            }
            PowerAnalysisControls { Layout.fillWidth: true; bridge: page.bridge; colors: page.colors; visible: page.state.analysisEnabled }
            Text { Layout.fillWidth: true; Layout.margins: 8; text: page.state.status; color: page.state.error ? page.colors.error : page.colors.muted; font.pixelSize: 11; wrapMode: Text.Wrap }
        }
    }
}
