pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot.refit : ({})
    objectName: "power_analysis_refit"
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "Manual refit of one Power spectrum"
    ColumnLayout {
        anchors.fill: parent
        spacing: 7
        Flow {
            Layout.fillWidth: true
            Layout.margins: 8
            spacing: 7
            AppComboBox {
                colors: page.colors
                objectName: "power_refit_channel"
                model: page.state.channels || []
                currentIndex: (page.state.channels || []).indexOf(page.state.channel)
                enabled: !page.state.busy
                Accessible.name: "Refit channel"
                onActivated: function(index) { page.bridge.setParameter("refit.channel", model[index]) }
            }
            AppComboBox {
                colors: page.colors
                objectName: "power_refit_model"
                model: ["Lorentzian", "Gaussian"]
                currentIndex: page.state.model === "Gaussian" ? 1 : 0
                enabled: !page.state.busy
                Accessible.name: "Manual peak model"
                onActivated: function(index) { page.bridge.setParameter("refit.model", model[index]) }
            }
            AppButton {
                objectName: "power_refit_run"
                text: "Refit selected spectrum"
                colors: page.colors
                enabled: !page.state.busy
                onClicked: page.bridge.invoke("refit.run")
            }
            AppButton {
                objectName: "power_refit_apply"
                text: "Apply this row"
                colors: page.colors
                primary: true
                enabled: page.state.canApply === true
                onClicked: page.bridge.invoke("refit.apply")
            }
            AppButton {
                objectName: "power_refit_cancel"
                text: "Cancel / Back"
                colors: page.colors
                onClicked: page.bridge.invoke("refit.cancel")
            }
        }
        Text {
            Layout.fillWidth: true
            Layout.leftMargin: 10
            Layout.rightMargin: 10
            text: (page.state.channel || "") + " · measured power " + Number(page.state.power || 0).toPrecision(9)
                + " uW · " + (page.state.source || "") + " · source row " + (page.state.sourceRow === undefined ? "—" : page.state.sourceRow)
                + ((page.state.origins || []).length ? "\nOrigins: " + page.state.origins.join(", ") : "")
            color: page.colors.text
            wrapMode: Text.Wrap
            Accessible.name: text
        }
        Text {
            Layout.fillWidth: true
            Layout.leftMargin: 10
            Layout.rightMargin: 10
            text: "Select a peak row, edit guesses and non-overlapping center bounds, then Refit to preview. Click the spectrum to move the selected center. Apply replaces this channel and measured row only."
            color: page.colors.muted
            font.pixelSize: 11
            wrapMode: Text.Wrap
            Accessible.name: text
        }
        Flickable {
            id: editor
            objectName: "power_refit_peak_editor"
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(235, 40 + (page.state.peaks || []).length * 43)
            contentWidth: Math.max(width, 900)
            contentHeight: 31 + (page.state.peaks || []).length * 43
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            Basic.ScrollBar.horizontal: Basic.ScrollBar { }
            Basic.ScrollBar.vertical: Basic.ScrollBar { }
            Column {
                width: editor.contentWidth
                Row {
                    height: 31
                    Text { width: 85; height: 31; text: "Peak"; color: page.colors.muted; verticalAlignment: Text.AlignVCenter; leftPadding: 8 }
                    Repeater {
                        model: ["Center (eV)", "FWHM (meV)", "Height guess", "Center min (eV)", "Center max (eV)"]
                        delegate: Text {
                            required property string modelData
                            width: 160; height: 31
                            text: modelData
                            color: page.colors.muted
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: 7
                            font.pixelSize: 11
                        }
                    }
                }
                Repeater {
                    model: page.state.peaks || []
                    delegate: Row {
                        id: peakRow
                        required property var modelData
                        required property int index
                        height: 43
                        spacing: 5
                        AppButton {
                            width: 80
                            text: "Peak " + (peakRow.index + 1)
                            colors: page.colors
                            checked: page.state.selectedPeak === peakRow.index
                            onClicked: page.bridge.setParameter("refit.selectedPeak", peakRow.index)
                        }
                        Repeater {
                            model: [{key: "center", title: "Center (eV)"}, {key: "width", title: "FWHM (meV)"},
                                {key: "height", title: "Height guess"}, {key: "lower", title: "Center min (eV)"}, {key: "upper", title: "Center max (eV)"}]
                            delegate: AppNumberField {
                                required property var modelData
                                width: 155
                                colors: page.colors
                                bridge: page.bridge
                                descriptor: ({key: "refit.peak." + peakRow.index + "." + modelData.key,
                                    label: "Peak " + (peakRow.index + 1) + " " + modelData.title,
                                    value: peakRow.modelData[modelData.key], min: modelData.key === "width" || modelData.key === "height" ? 0.000001 : -10000,
                                    max: modelData.key === "height" ? 1e15 : 10000, decimals: 7, step: modelData.key === "center" ? 0.001 : 0.1,
                                    displayDecimals: modelData.key === "height" ? undefined : modelData.key === "width" ? 3 : 6,
                                    enabled: !page.state.busy})
                            }
                        }
                    }
                }
            }
        }
        PlotSurface {
            objectName: "power_refit_plot"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 140
            backend: page.bridge ? page.bridge.plotBackend : null
        }
        Text {
            objectName: "power_refit_status"
            Layout.fillWidth: true
            Layout.margins: 9
            text: page.state.status || ""
            color: page.colors.muted
            wrapMode: Text.Wrap
            Accessible.name: text
        }
    }
}
