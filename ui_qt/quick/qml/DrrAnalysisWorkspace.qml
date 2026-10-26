pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    color: colors.inset
    objectName: "drr_analysis_workspace"
    ColumnLayout {
        anchors.fill: parent
        spacing: 4
        RowLayout {
            Layout.fillWidth: true
            Basic.TabBar {
                Layout.fillWidth: true
                currentIndex: page.state.values.tab
                AppTabButton { colors: page.colors; text: "Peak positions"; onClicked: page.bridge.setParameter("tab", 0) }
                AppTabButton { colors: page.colors; text: "Peak-to-peak"; onClicked: page.bridge.setParameter("tab", 1) }
            }
            AppButton { colors: page.colors; text: "Home"; onClicked: page.bridge.invoke("home") }
            AppButton { colors: page.colors; text: "Back"; onClicked: page.bridge.invoke("back") }
            AppButton { colors: page.colors; text: "Pan"; primary: page.state.values.plotTool === "pan"; onClicked: page.bridge.invoke("pan") }
            AppButton { colors: page.colors; text: "Zoom"; primary: page.state.values.plotTool === "zoom"; onClicked: page.bridge.invoke("zoom") }
        }
        Basic.SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Vertical
            PlotSurface {
                objectName: "drr_analysis_plot"
                Basic.SplitView.fillWidth: true
                Basic.SplitView.fillHeight: true
                Basic.SplitView.minimumHeight: 160
                backend: page.bridge.plotBackend
                Text {
                    anchors.centerIn: parent
                    visible: !page.state.activeKey
                    text: "Add DRR datasets to begin"
                    color: page.colors.muted
                }
            }
            ColumnLayout {
                visible: page.state.values.tab === 0
                Basic.SplitView.preferredHeight: visible ? 205 : 0
                Basic.SplitView.minimumHeight: visible ? 80 : 0
                spacing: 2
                RowLayout {
                    Layout.fillWidth: true
                    Basic.Label {
                        Layout.fillWidth: true
                        text: page.state.candidates + " candidates → " + page.state.retained + " retained · Page " + page.state.page + "/" + page.state.pages
                        color: page.colors.muted
                    }
                    AppButton { colors: page.colors; text: "Previous"; enabled: page.state.page > 1; onClicked: page.bridge.invoke("previous") }
                    AppButton { colors: page.colors; text: "Next"; enabled: page.state.page < page.state.pages; onClicked: page.bridge.invoke("next") }
                }
                Flickable {
                    id: tableScroll
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contentWidth: Math.max(width, 850)
                    contentHeight: height
                    clip: true
                    Basic.ScrollBar.horizontal: Basic.ScrollBar {}
                    ColumnLayout {
                        width: tableScroll.contentWidth
                        height: tableScroll.height
                        spacing: 0
                        Row {
                            Layout.fillWidth: true
                            Repeater {
                                model: ["Source", "Y", "Track", "Type", "Energy (eV)", "Status", "Detection prom.", "Width (meV)", "Auto support"]
                                delegate: Text { required property string modelData; width: tableScroll.contentWidth / 9; height: 25; text: modelData; color: page.colors.muted; font.pixelSize: 11; verticalAlignment: Text.AlignVCenter }
                            }
                        }
                        ListView {
                            id: results
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            model: page.bridge.peakResultsModel
                            clip: true
                            Basic.ScrollBar.vertical: Basic.ScrollBar {}
                            delegate: Rectangle {
                                required property var entry
                                required property int index
                                width: results.width
                                height: 27
                                color: entry.selected ? page.colors.selected : (index % 2 ? page.colors.surface : page.colors.inset)
                                Row {
                                    anchors.fill: parent
                                    Repeater {
                                        model: entry.cells
                                        delegate: Text { required property string modelData; width: results.width / 9; height: 27; text: modelData; color: page.colors.text; font.pixelSize: 11; elide: Text.ElideRight; verticalAlignment: Text.AlignVCenter }
                                    }
                                }
                                MouseArea { anchors.fill: parent; onClicked: page.bridge.selectPeak(entry.index) }
                            }
                        }
                    }
                }
            }
        }
        Basic.Label {
            Layout.fillWidth: true
            visible: page.state.values.tab === 1 && page.state.p2p.recordCount !== undefined
            text: page.state.p2p.recordCount + " files · " + page.state.p2p.edges + " rows near window edges · " + page.state.p2p.missing + " missing rows" + (page.state.p2p.pending ? " · calculation pending" : "")
            color: page.colors.muted
            wrapMode: Text.Wrap
        }
    }
}
