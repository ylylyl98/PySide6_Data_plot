pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: panel
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    objectName: "power_analysis_results"
    visible: state.analysisEnabled === true
    implicitHeight: visible ? (state.analysisShowResults ? 245 : actions.implicitHeight + summary.implicitHeight + 22) : 0
    color: colors.surface
    Accessible.role: Accessible.Pane
    Accessible.name: "Power peak-fit results and provenance"
    function number(value) { return value === null || value === undefined ? "—" : Number(value).toPrecision(6) }
    ColumnLayout {
        anchors.fill: parent
        spacing: 4
        Flow {
            id: actions
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 8
            Layout.topMargin: 5
            spacing: 5
            AppButton {
                objectName: "power_analysis_manual"
                colors: panel.colors
                text: "Manual refit selected power…"
                enabled: panel.state.analysisReady === true && !panel.state.fitBusy
                onClicked: panel.bridge.invoke("analysis.manual")
            }
            AppButton {
                objectName: "power_analysis_law"
                colors: panel.colors
                text: "Fit power law…"
                enabled: panel.state.analysisReady === true && !panel.state.fitBusy
                onClicked: panel.bridge.invoke("analysis.law")
            }
            AppButton {
                objectName: "power_analysis_toggle_results"
                colors: panel.colors
                text: "Results table"
                checked: panel.state.analysisShowResults === true
                onClicked: panel.bridge.setParameter("analysis.showResults", !panel.state.analysisShowResults)
            }
        }
        Text {
            id: summary
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 8
            Layout.topMargin: 6
            text: panel.state.analysisLawWarning || panel.state.analysisStatus || ""
            color: panel.colors.muted
            font.pixelSize: 11
            wrapMode: Text.Wrap
            Accessible.name: text
        }
        Flickable {
            id: viewport
            visible: panel.state.analysisShowResults === true
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: Math.max(width, 1180)
            contentHeight: height
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            Basic.ScrollBar.horizontal: Basic.ScrollBar { }
            ListView {
                id: results
                width: viewport.contentWidth
                height: viewport.height
                clip: true
                model: panel.bridge ? panel.bridge.peakResultsModel : null
                Basic.ScrollBar.vertical: Basic.ScrollBar { }
                headerPositioning: ListView.OverlayHeader
                header: Rectangle {
                    width: results.width
                    height: 28
                    z: 2
                    color: panel.colors.surface
                    Row {
                        Repeater {
                            model: [{text: "Channel / peak", size: 120}, {text: "Power (uW)", size: 105}, {text: "Area (a.u. eV)", size: 115},
                                {text: "Height (a.u.)", size: 110}, {text: "FWHM ± SE (meV)", size: 150},
                                {text: "Center (eV)", size: 110}, {text: "Status / manual", size: 185}, {text: "Source / row", size: 285}]
                            delegate: Text {
                                required property var modelData
                                width: modelData.size; height: 28
                                text: modelData.text; color: panel.colors.muted
                                font.pixelSize: 11; leftPadding: 7; verticalAlignment: Text.AlignVCenter
                            }
                        }
                    }
                }
                delegate: Basic.ItemDelegate {
                    id: fitRow
                    required property var entry
                    readonly property var modelData: entry
                    required property int index
                    width: results.width; height: 31; padding: 0
                    Accessible.name: modelData.channel + " Peak " + modelData.peak + ", power " + panel.number(modelData.power)
                        + ", " + modelData.status + ", source " + modelData.source + ", row " + modelData.sourceRow
                    Accessible.description: "Inspect this measured spectrum and retain its exact channel and source row for manual refit"
                        + "; origins: " + modelData.sources.join(", ")
                    Basic.ToolTip.visible: hovered
                    Basic.ToolTip.text: "Origins: " + modelData.sources.join(", ") + "\nSource row: " + modelData.sourceRow
                    Basic.ToolTip.delay: 600
                    background: Rectangle {
                        color: fitRow.down ? panel.colors.pressed : fitRow.hovered ? panel.colors.hover
                            : fitRow.index % 2 ? panel.colors.surface : panel.colors.inset
                        border.width: fitRow.visualFocus ? 2 : 0
                        border.color: panel.colors.focus
                    }
                    contentItem: Row {
                        Repeater {
                            model: [{text: fitRow.modelData.channel + " / Peak " + fitRow.modelData.peak, size: 120},
                                {text: panel.number(fitRow.modelData.power), size: 105}, {text: panel.number(fitRow.modelData.area), size: 115},
                                {text: panel.number(fitRow.modelData.height), size: 110},
                                {text: panel.number(fitRow.modelData.fwhm) + " ± " + panel.number(fitRow.modelData.fwhmError), size: 150},
                                {text: panel.number(fitRow.modelData.center), size: 110},
                                {text: fitRow.modelData.status + (fitRow.modelData.manual ? " · manual" : ""), size: 185},
                                {text: fitRow.modelData.source + " / " + fitRow.modelData.sourceRow, size: 285}]
                            delegate: Text {
                                required property var modelData
                                width: modelData.size; height: 31
                                text: modelData.text; color: fitRow.modelData.status === "ok" ? panel.colors.text : panel.colors.muted
                                font.pixelSize: 11; leftPadding: 7; verticalAlignment: Text.AlignVCenter; elide: Text.ElideRight
                            }
                        }
                    }
                    onClicked: panel.bridge.setParameter("analysis.selectedResult", index)
                }
            }
        }
    }
}
