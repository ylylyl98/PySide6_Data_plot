pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: panel
    required property var bridge
    required property var colors
    readonly property var state: bridge.snapshot
    color: colors.surface
    objectName: "peak_analysis_results"
    function number(value) { return value === undefined || value === null ? "—" : Number(value).toPrecision(6) }
    function cells(point) {
        return [point.branch, point.row_index, number(point.y), number(point.energy), point.polarity, point.status,
            number(point.snr), number(point.width_mev), number(point.fit_center), point.fit_status || "—",
            number(point.fwhm_mev), number(point.height), number(point.area), number(point.rms)]
    }
    ColumnLayout {
        anchors.fill: parent; spacing: 2
        RowLayout {
            Layout.fillWidth: true; Layout.leftMargin: 8; Layout.rightMargin: 8
            Basic.Label { Layout.fillWidth: true; text: "Results · " + panel.state.resultCount + " positions · " + panel.state.resultMethod; color: panel.colors.text }
            AppButton { text: "Close"; colors: panel.colors; onClicked: panel.bridge.invoke("close-results") }
        }
        Basic.ScrollView {
            id: scroll
            Layout.fillWidth: true; Layout.fillHeight: true
            contentWidth: 1540
            clip: true
            ListView {
                id: results
                width: Math.max(scroll.availableWidth, 1540)
                height: scroll.availableHeight
                model: panel.bridge.resultsModel
                clip: true
                headerPositioning: ListView.OverlayHeader
                header: Rectangle {
                    z: 2; width: results.width; height: 31; color: panel.colors.control
                    Row {
                        Repeater {
                            model: ["Branch", "Row", panel.state.scanLabel || "Scan", "Energy (eV)", "Polarity", "Status", "SNR", "Width (meV)", "Fit center", "Fit status", "FWHM (meV)", "Height", "Area", "RMS"]
                            delegate: Text {
                                required property string modelData
                                width: 110; height: 31; leftPadding: 7
                                text: modelData; color: panel.colors.muted
                                verticalAlignment: Text.AlignVCenter; elide: Text.ElideRight; font.pixelSize: 11
                            }
                        }
                    }
                }
                delegate: Basic.ItemDelegate {
                    id: pointRow
                    required property var entry
                    required property int index
                    width: results.width; height: 29; padding: 0
                    Accessible.name: entry.branch + ", row " + entry.row_index + ", energy " + panel.number(entry.energy) + ", " + entry.status
                    onClicked: panel.bridge.setParameter("resultRow", index)
                    background: Rectangle {
                        color: pointRow.entry.row_index === panel.state.row ? panel.colors.selected
                            : pointRow.hovered ? panel.colors.hover : pointRow.index % 2 ? panel.colors.surface : panel.colors.inset
                        border.width: pointRow.visualFocus ? 2 : 0; border.color: panel.colors.focus
                    }
                    contentItem: Row {
                        Repeater {
                            model: panel.cells(pointRow.entry)
                            delegate: Text {
                                required property var modelData
                                width: 110; height: 29; leftPadding: 7
                                text: String(modelData); color: panel.colors.text
                                verticalAlignment: Text.AlignVCenter; elide: Text.ElideRight; font.pixelSize: 11
                            }
                        }
                    }
                }
            }
        }
    }
}
