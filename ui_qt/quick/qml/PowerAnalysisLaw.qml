pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: page
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot.law : ({})
    objectName: "power_analysis_law"
    property bool advancedOpen: false
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "Power law intervals and local slopes"
    ColumnLayout {
        anchors.fill: parent
        spacing: 7
        Flow {
            Layout.fillWidth: true
            Layout.margins: 8
            spacing: 8
            Basic.CheckBox {
                objectName: "power_law_shared"
                text: "Shared fitting range"
                checked: page.state.shared === true
                onClicked: page.bridge.setParameter("law.shared", checked)
                Accessible.name: text
            }
            AppButton {
                objectName: "power_law_keep"
                text: "Keep interval result"
                colors: page.colors
                enabled: page.state.canApply === true && (page.state.rows || []).some(function(row) { return row.selected })
                onClicked: page.bridge.invoke("law.keep")
            }
            AppButton {
                objectName: "power_law_advanced"
                text: page.advancedOpen ? "Hide advanced" : "Advanced"
                colors: page.colors
                onClicked: page.advancedOpen = !page.advancedOpen
            }
            AppButton {
                objectName: "power_law_apply"
                text: "Apply power law"
                colors: page.colors
                primary: true
                enabled: page.state.canApply === true
                onClicked: page.bridge.invoke("law.apply")
            }
            AppButton {
                objectName: "power_law_cancel"
                text: "Back"
                colors: page.colors
                onClicked: page.bridge.invoke("law.cancel")
            }
        }
        Flow {
            Layout.fillWidth: true; Layout.leftMargin: 8; Layout.rightMargin: 8
            spacing: 8
            visible: page.advancedOpen
            Basic.CheckBox {
                objectName: "power_law_local"
                text: "Show local slope"
                checked: page.state.local === true
                onClicked: page.bridge.setParameter("law.local", checked)
                Accessible.name: text
            }
            Basic.ComboBox {
                objectName: "power_law_area_metric"
                implicitWidth: 210
                model: ["Area · measured range", "Area · full fitted peak"]
                currentIndex: page.state.areaMetric === "full" ? 1 : 0
                Accessible.name: "Power law area definition"
                onActivated: function(index) { page.bridge.setParameter("law.areaMetric", index === 1 ? "full" : "legacy") }
            }
            AppButton {
                objectName: "power_law_suggest"
                text: "Suggest ranges near alpha = 1"
                colors: page.colors
                enabled: !page.state.busy
                onClicked: page.bridge.invoke("law.suggest")
            }
            Text { width: Math.min(300, parent.width); text: "Measured range preserves earlier results. The suggestion is exploratory; review the interval."; color: page.colors.muted; wrapMode: Text.Wrap }
        }
        Text {
            Layout.fillWidth: true; Layout.leftMargin: 9; Layout.rightMargin: 9
            visible: (page.state.history || []).length > 0
            text: "Kept intervals: " + (page.state.history || []).map(function(fit) {
                return fit.name + " [" + Number(fit.lower).toPrecision(4) + ", " + Number(fit.upper).toPrecision(4) + "] α=" + Number(fit.alpha).toPrecision(4) + " (" + (fit.intensity_metric || "area") + ")"
            }).join("; ")
            color: page.colors.muted; wrapMode: Text.Wrap; elide: Text.ElideRight; maximumLineCount: 2
        }
        Text {
            Layout.fillWidth: true
            Layout.leftMargin: 9
            Layout.rightMargin: 9
            text: (page.state.areaMetric === "full" ? "Area · full fitted peak. " : "Area · measured range. ") + "Select peaks, then enter limits or drag the interval on the plot. Dragging adjusts the highlighted row, or all selected peaks when ranges are shared. Limits are independent of the plot axis bounds."
            color: page.colors.muted
            wrapMode: Text.Wrap
            font.pixelSize: 11
            Accessible.name: text
        }
        Flickable {
            id: editor
            objectName: "power_law_interval_editor"
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(240, 31 + (page.state.rows || []).length * 66)
            contentWidth: Math.max(width, 900)
            contentHeight: 31 + (page.state.rows || []).length * 66
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            Basic.ScrollBar.horizontal: Basic.ScrollBar {
                objectName: "power_law_interval_horizontal_scrollbar"
                visible: editor.contentWidth > editor.width
                policy: editor.contentWidth > editor.width ? Basic.ScrollBar.AlwaysOn : Basic.ScrollBar.AsNeeded
            }
            Basic.ScrollBar.vertical: Basic.ScrollBar { }
            Column {
                width: editor.contentWidth
                Row {
                    height: 31
                    Repeater {
                        model: [{text: "Fit / highlighted peak", width: 190}, {text: "Min power (uW)", width: 155},
                            {text: "Max power (uW)", width: 155}, {text: "Result · alpha ± 1 standard error", width: 390}]
                        delegate: Text {
                            required property var modelData
                            width: modelData.width
                            height: 31
                            text: modelData.text
                            color: page.colors.muted
                            font.pixelSize: 11
                            leftPadding: 7
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
                Repeater {
                    model: page.state.rows || []
                    delegate: Rectangle {
                        id: interval
                        required property var modelData
                        required property int index
                        width: editor.contentWidth
                        height: 66
                        color: page.state.selectedRow === index ? page.colors.selected : page.colors.inset
                        Row {
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: 5
                            Basic.CheckBox {
                                objectName: "power_law_select_" + interval.index
                                width: 35
                                checked: interval.modelData.selected
                                onClicked: page.bridge.setParameter("law." + interval.index + ".selected", checked)
                                Accessible.name: "Fit " + interval.modelData.name
                            }
                            AppButton {
                                width: 145
                                text: interval.modelData.name
                                colors: page.colors
                                checked: page.state.selectedRow === interval.index
                                description: "Highlight " + interval.modelData.name + " for interval dragging"
                                onClicked: page.bridge.setParameter("law.selectedRow", interval.index)
                            }
                            Repeater {
                                model: ["lower", "upper"]
                                delegate: AppNumberField {
                                    required property string modelData
                                    width: 150
                                    bridge: page.bridge
                                    colors: page.colors
                                    descriptor: ({key: "law." + interval.index + "." + modelData,
                                        label: interval.modelData.name + (modelData === "lower" ? " minimum power" : " maximum power"),
                                        value: interval.modelData[modelData], min: 1e-10, max: 1e15, step: 0.1, decimals: 10})
                                }
                            }
                            Text {
                                width: 380
                                text: interval.modelData.result || (page.state.busy && interval.modelData.selected ? "Fitting interval…" : "")
                                color: interval.modelData.error ? page.colors.error : page.colors.text
                                font.pixelSize: 11
                                wrapMode: Text.Wrap
                                Accessible.name: interval.modelData.name + ": " + text
                            }
                        }
                    }
                }
            }
        }
        PlotSurface {
            objectName: "power_law_plot"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 140
            backend: page.bridge ? page.bridge.plotBackend : null
        }
        Text {
            objectName: "power_law_message"
            Layout.fillWidth: true
            Layout.margins: 9
            text: !page.advancedOpen && (page.state.message || "").indexOf("Selecting intervals near alpha") === 0
                ? "Error = 1 fit standard error." : page.state.message || "Error = 1 fit standard error."
            color: page.colors.muted
            font.pixelSize: 11
            wrapMode: Text.Wrap
            Accessible.name: text
        }
    }
}
