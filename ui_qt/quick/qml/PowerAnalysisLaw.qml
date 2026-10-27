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
            Basic.CheckBox {
                objectName: "power_law_local"
                text: "Show local slope"
                checked: page.state.local === true
                onClicked: page.bridge.setParameter("law.local", checked)
                Accessible.name: text
            }
            AppButton {
                objectName: "power_law_suggest"
                text: "Suggest ranges near alpha = 1"
                colors: page.colors
                enabled: !page.state.busy
                onClicked: page.bridge.invoke("law.suggest")
            }
            AppButton {
                objectName: "power_law_apply"
                text: "Apply intervals"
                colors: page.colors
                primary: true
                enabled: page.state.canApply === true
                onClicked: page.bridge.invoke("law.apply")
            }
            AppButton {
                objectName: "power_law_cancel"
                text: "Cancel / Back"
                colors: page.colors
                onClicked: page.bridge.invoke("law.cancel")
            }
        }
        Text {
            Layout.fillWidth: true
            Layout.leftMargin: 9
            Layout.rightMargin: 9
            text: "Select peaks, then enter limits or drag the interval on the plot. Dragging adjusts the highlighted row, or all selected peaks when ranges are shared. Limits are independent of the plot axis bounds."
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
            Basic.ScrollBar.horizontal: Basic.ScrollBar { }
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
            text: page.state.message || ""
            color: page.colors.muted
            font.pixelSize: 11
            wrapMode: Text.Wrap
            Accessible.name: text
        }
    }
}
