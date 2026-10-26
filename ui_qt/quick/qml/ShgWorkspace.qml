pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    objectName: "quick_shg_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({mode: "single", busy: false,
        inspectionVisible: false, showTable: false, canExport: false, sources: {}, rows: [],
        workspaceMessage: "", fitSummary: "", fitMessage: "", summary: "", status: "", error: "",
        compactInspection: false, spectrumPanel: "raw"})
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "SHG spectra, angular fit and twist comparison"
    onColorsChanged: if (bridge) bridge.setPalette(colors)
    Component.onCompleted: if (bridge) bridge.setPalette(colors)

    function value(number) { return number === null || number === undefined ? "—" : Number(number).toPrecision(7) }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Flow {
            Layout.fillWidth: true
            Layout.margins: 7
            spacing: 4
            AppButton {
                objectName: "shg_single"
                colors: page.colors
                text: "Single file"
                primary: page.state.mode === "single"
                enabled: !page.state.busy
                onClicked: page.bridge.setParameter("mode", "single")
            }
            AppButton {
                objectName: "shg_compare"
                colors: page.colors
                text: "Compare / Twist"
                primary: page.state.mode === "compare"
                enabled: !page.state.busy
                onClicked: page.bridge.setParameter("mode", "compare")
            }
            AppButton {
                objectName: "shg_reload"
                colors: page.colors
                text: "Reload"
                enabled: !page.state.busy && Object.keys(page.state.sources || {}).length > 0
                onClicked: page.bridge.invoke("reload")
            }
            AppButton {
                objectName: "shg_table"
                colors: page.colors
                text: "Results table"
                checked: page.state.showTable
                enabled: page.state.inspectionVisible
                onClicked: page.bridge.invoke("toggle-table")
            }
            AppButton {
                objectName: "shg_raw_panel"
                colors: page.colors
                text: "Raw spectrum"
                visible: page.state.mode === "single" && page.state.compactInspection
                checked: page.state.spectrumPanel === "raw"
                onClicked: page.bridge.setParameter("spectrumPanel", "raw")
            }
            AppButton {
                objectName: "shg_corrected_panel"
                colors: page.colors
                text: "Corrected spectrum"
                visible: page.state.mode === "single" && page.state.compactInspection
                checked: page.state.spectrumPanel === "corrected"
                onClicked: page.bridge.setParameter("spectrumPanel", "corrected")
            }
            AppButton {
                objectName: "shg_cancel"
                colors: page.colors
                text: "Cancel"
                visible: page.state.busy
                onClicked: page.bridge.invoke("cancel")
            }
        }
        Basic.SplitView {
            objectName: "shg_workspace_split"
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Vertical
            handle: Rectangle {
                implicitHeight: 5
                color: Basic.SplitHandle.pressed ? page.colors.accent : page.colors.border
            }
            Item {
                Basic.SplitView.fillHeight: true
                Basic.SplitView.minimumHeight: 170
                PlotSurface {
                    objectName: "shg_plot_surface"
                    anchors.fill: parent
                    backend: page.bridge ? page.bridge.plotBackend : null
                    visible: page.state.inspectionVisible === true
                }
                Text {
                    objectName: "shg_workspace_message"
                    anchors.centerIn: parent
                    width: Math.max(80, Math.min(520, parent.width - 40))
                    visible: !page.state.inspectionVisible
                    text: page.state.workspaceMessage || ""
                    color: page.state.error ? page.colors.error : page.colors.muted
                    font.pixelSize: 14
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                    Accessible.name: text
                }
            }
            Flickable {
                id: tableViewport
                objectName: "shg_results_table"
                visible: page.state.showTable && page.state.inspectionVisible
                Basic.SplitView.preferredHeight: 180
                Basic.SplitView.minimumHeight: visible ? 100 : 0
                Basic.SplitView.maximumHeight: visible ? 400 : 0
                clip: true
                contentWidth: Math.max(width, 850)
                contentHeight: height
                boundsBehavior: Flickable.StopAtBounds
                Basic.ScrollBar.horizontal: Basic.ScrollBar { }
                ListView {
                    id: rows
                    width: tableViewport.contentWidth
                    height: tableViewport.height
                    clip: true
                    model: page.state.rows || []
                    boundsBehavior: Flickable.StopAtBounds
                    Basic.ScrollBar.vertical: Basic.ScrollBar { }
                    headerPositioning: ListView.OverlayHeader
                    header: Rectangle {
                        width: rows.width
                        height: 30
                        z: 2
                        color: page.colors.surface
                        Row {
                            anchors.fill: parent
                            Repeater {
                                model: [{title: "Dataset / row", size: 140}, {title: "Angle (°)", size: 105},
                                    {title: "Area (counts·nm)", size: 145}, {title: "Uncertainty", size: 125},
                                    {title: "Included / fit", size: 115}, {title: "Cosmic / quality", size: 220}]
                                delegate: Text {
                                    required property var modelData
                                    width: modelData.size
                                    height: 30
                                    leftPadding: 8
                                    text: modelData.title
                                    color: page.colors.muted
                                    verticalAlignment: Text.AlignVCenter
                                    font.pixelSize: 11
                                }
                            }
                        }
                    }
                    delegate: Basic.ItemDelegate {
                        id: row
                        required property var modelData
                        required property int index
                        width: rows.width
                        height: 32
                        padding: 0
                        Accessible.name: modelData.dataset + ", source row " + modelData.sourceRow
                            + ", angle " + page.value(modelData.angle) + ", integrated area " + page.value(modelData.area)
                            + ", " + (modelData.included ? "included" : "excluded") + ", " + modelData.quality
                        Accessible.description: "Select to inspect the nearest measured-angle spectrum in single-file mode"
                        background: Rectangle {
                            color: row.down ? page.colors.pressed : row.hovered ? page.colors.hover
                                : row.index % 2 ? page.colors.surface : page.colors.inset
                            border.width: row.visualFocus ? 2 : 0
                            border.color: page.colors.focus
                        }
                        contentItem: Row {
                            Repeater {
                                model: [{text: row.modelData.dataset + " / " + row.modelData.sourceRow, size: 140},
                                    {text: page.value(row.modelData.angle), size: 105}, {text: page.value(row.modelData.area), size: 145},
                                    {text: page.value(row.modelData.uncertainty), size: 125},
                                    {text: (row.modelData.included ? "Yes" : "No") + " / " + (row.modelData.fitIncluded ? "Yes" : "No"), size: 115},
                                    {text: row.modelData.cosmicPixels + " / " + row.modelData.quality, size: 220}]
                                delegate: Text {
                                    required property var modelData
                                    text: modelData.text
                                    width: modelData.size
                                    height: 32
                                    leftPadding: 8
                                    elide: Text.ElideRight
                                    color: row.modelData.included ? page.colors.text : page.colors.muted
                                    verticalAlignment: Text.AlignVCenter
                                    font.pixelSize: 11
                                }
                            }
                        }
                        onClicked: if (page.state.mode === "single" && modelData.angle !== null)
                            page.bridge.setParameter("angleCursor", modelData.angle)
                    }
                }
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: 8
            spacing: 3
            Text {
                objectName: "shg_fit_summary"
                Layout.fillWidth: true
                text: page.state.fitSummary || ""
                visible: text.length > 0
                color: page.colors.text
                font.pixelSize: 12
                wrapMode: Text.Wrap
                Accessible.name: text
            }
            Text {
                objectName: "shg_status"
                Layout.fillWidth: true
                text: page.state.fitMessage || page.state.summary || page.state.status || ""
                color: page.state.error ? page.colors.error : page.colors.muted
                font.pixelSize: 11
                wrapMode: Text.Wrap
                Accessible.name: text
            }
        }
    }
}
