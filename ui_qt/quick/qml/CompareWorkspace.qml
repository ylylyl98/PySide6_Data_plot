pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    objectName: "quick_compare_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "Compare maps and spectra"

    onColorsChanged: bridge.setPalette(colors)
    Component.onCompleted: bridge.setPalette(colors)

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Flow {
            Layout.fillWidth: true
            Layout.margins: 8
            spacing: 5
            AppButton {
                objectName: "compare_intensity"
                colors: page.colors
                text: "Intensity"
                primary: page.state.view === "intensity"
                onClicked: page.bridge.setParameter("view", "intensity")
            }
            AppButton {
                objectName: "compare_vp"
                colors: page.colors
                text: "VP"
                description: "Compare KK and KKp valley polarization"
                primary: page.state.view === "vp"
                onClicked: page.bridge.setParameter("view", "vp")
            }
            AppButton {
                objectName: "compare_reload"
                colors: page.colors
                text: "Reload"
                enabled: !page.state.busy && Object.keys(page.state.assignments || {}).length >= 2
                onClicked: page.bridge.invoke("reload")
            }
        }
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            PlotSurface {
                id: plot
                objectName: "compare_plot_surface"
                anchors.fill: parent
                backend: page.bridge.plotBackend
                visible: page.state.inspectionVisible
            }
            AppInspectionDivider {
                anchors.fill: parent
                visible: page.state.inspectionVisible
                bridge: page.bridge
                state: page.state
                colors: page.colors
            }
            Text {
                objectName: "compare_workspace_message"
                anchors.centerIn: parent
                width: Math.min(500, parent.width - 48)
                visible: !page.state.inspectionVisible
                text: page.state.busy ? "Loading Compare channels…" : page.state.workspaceMessage
                color: page.state.error ? page.colors.error : page.colors.muted
                font.pixelSize: 15
                wrapMode: Text.Wrap
                horizontalAlignment: Text.AlignHCenter
                Accessible.name: text
            }
        }
        Text {
            objectName: "compare_status"
            Layout.fillWidth: true
            Layout.margins: 8
            text: page.state.status
            color: page.state.error ? page.colors.error : page.colors.muted
            font.pixelSize: 11
            wrapMode: Text.Wrap
            Accessible.name: text
        }
    }
}
