pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window
import Dptk 1.0
import "NumericFormat.js" as NumericFormat

Rectangle {
    id: page
    objectName: "quick_megasweep_workspace"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    color: colors.inset
    Accessible.role: Accessible.Pane
    Accessible.name: "Megasweep plot and background references"
    onColorsChanged: if (bridge) bridge.setPalette(colors)
    Component.onCompleted: if (bridge) bridge.setPalette(colors)

    MegasweepMovieDialog { id: movieDialog; parent: page.Window.window ? page.Window.window.contentItem : page; bridge: page.bridge; colors: page.colors }
    component ToolbarButton: AppButton {
        colors: page.colors
        compact: true
        implicitHeight: 26
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Flow {
            Layout.fillWidth: true
            Layout.margins: 4
            spacing: 3
            ToolbarButton { text: "Leave figure edit"; visible: page.state.savedFigureActive === true; onClicked: page.bridge.invoke("close-slide-figure") }
            ToolbarButton { objectName: "megasweep_return_to_map"; text: "Analysis map"; visible: page.state.mode === "Reflection"; checked: !page.state.reflectionPreviewVisible && page.state.values?.view !== "slice"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: if (!checked) page.bridge.invoke("return-to-map") }
            ToolbarButton { objectName: "megasweep_reflection_preview"; text: "Reference spectra"; visible: page.state.mode === "Reflection"; checked: page.state.reflectionPreviewVisible === true; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: if (!page.state.reflectionPreviewVisible) page.bridge.invoke("reflection-preview") }
            ToolbarButton { objectName: "megasweep_spectral_cut"; text: "Spectral cut"; visible: page.state.mode === "Reflection"; checked: !page.state.reflectionPreviewVisible && page.state.values?.view === "slice"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: if (!checked) page.bridge.invoke("show-spectral-cut") }
            ToolbarButton { objectName: "megasweep_slice_previous"; text: "‹ Slice"; description: "Previous slice"; visible: page.state.values && page.state.values.view === "slice" && !page.state.reflectionPreviewVisible; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("slice-previous") }
            ToolbarButton { objectName: "megasweep_slice_next"; text: "Slice ›"; description: "Next slice"; visible: page.state.values && page.state.values.view === "slice" && !page.state.reflectionPreviewVisible; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("slice-next") }
            ToolbarButton { objectName: "megasweep_export_batch"; text: "Batch export"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: page.bridge.invoke("export-batch") }
            ToolbarButton { objectName: "megasweep_export_video"; text: "Export video…"; visible: page.state.mode !== "Transport"; enabled: !page.state.busy && !!page.state.sourceFile; onClicked: movieDialog.open() }
            ToolbarButton { objectName: "megasweep_cancel"; text: "Cancel"; visible: page.state.busy; onClicked: page.bridge.invoke("cancel") }
            RowLayout {
                spacing: 3
                visible: page.state.mode === "Reflection" && (page.state.values?.view !== "slice" || page.state.reflectionPreviewVisible === true)
                Basic.Label { text: "Map quantity"; color: page.colors.text }
                AppComboBox {
                    id: quantity
                    objectName: "megasweep_map_quantity"
                    Layout.preferredWidth: 174
                    implicitHeight: 26
                    colors: page.colors
                    readonly property var options: (page.state.parameterGroups || []).find(g => g.key === "data")?.fields.find(f => f.key === "view")?.options.filter(o => o.value !== "slice") || []
                    model: options.map(o => o.label)
                    readonly property string metric: page.state.values?.view === "slice" ? (page.state.values.last_map_view || "rc_p2p") : (page.state.values?.view || "")
                    currentIndex: options.findIndex(o => o.value === metric)
                    enabled: !page.state.busy
                    Accessible.name: "Map quantity"
                    onActivated: index => page.bridge.setParameter("view", options[index].value)
                }
            }
        }
        MegasweepMapLimits {
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 8
            bridge: page.bridge
            colors: page.colors
            active: !!page.state.inspectionVisible && !page.state.reflectionPreviewVisible && !page.state.error
            enabled: !page.state.busy
        }
        Flow {
            id: referencePicker
            readonly property var point: page.state.referencePoint || ({})
            visible: page.state.reflectionPreviewVisible === true
            Layout.fillWidth: true
            Layout.leftMargin: 8
            spacing: 6
            Repeater {
                model: ["x", "y"]
                delegate: RowLayout {
                    id: coordinate
                    required property string modelData
                    readonly property string axis: modelData
                    Basic.Label { text: coordinate.axis.toUpperCase() + " · " + (referencePicker.point[coordinate.axis + "Name"] || ""); color: page.colors.text }
                    Basic.TextField {
                        objectName: "megasweep_reference_" + coordinate.axis
                        Layout.preferredWidth: 95
                        selectByMouse: true
                        inputMethodHints: Qt.ImhFormattedNumbersOnly
                        Accessible.name: "Sample " + coordinate.axis.toUpperCase() + " " + (referencePicker.point[coordinate.axis + "Name"] || "")
                        Accessible.description: "Enter a coordinate to preview the nearest measured value."
                        readonly property string measured: NumericFormat.exact(referencePicker.point[coordinate.axis])
                        property bool edited: false
                        onMeasuredChanged: if (!edited) text = measured
                        Component.onCompleted: text = measured
                        onTextEdited: edited = true
                        onEditingFinished: {
                            if (!edited) return
                            let value = NumericFormat.parse(text)
                            edited = false
                            if (value !== undefined && referencePicker.visible)
                                page.bridge.setReferenceCoordinate(coordinate.axis, value)
                            text = measured
                        }
                        Keys.onEscapePressed: { edited = false; text = measured }
                    }
                }
            }
            RowLayout {
                visible: (referencePicker.point.repeats || 0) > 1
                Basic.Label { text: "Repeat"; color: page.colors.text }
                Basic.SpinBox {
                    from: 1
                    to: Math.max(1, referencePicker.point.repeats || 0)
                    value: referencePicker.point.repeat || 1
                    onValueModified: page.bridge.setReferencePoint(referencePicker.point.x, referencePicker.point.y, value)
                }
            }
        }
        Basic.SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Vertical
            handle: Rectangle { implicitHeight: 5; color: Basic.SplitHandle.pressed ? page.colors.accent : page.colors.border }
            Item {
                Basic.SplitView.fillHeight: true
                Basic.SplitView.minimumHeight: 160
                PlotSurface { objectName: "megasweep_plot_surface"; anchors.fill: parent; backend: page.bridge ? page.bridge.plotBackend : null }
                Text {
                    objectName: "megasweep_workspace_message"
                    anchors.centerIn: parent
                    width: Math.max(80, Math.min(520, parent.width - 40))
                    visible: !page.state.sourceFile || (!!page.state.error && !page.state.canExport)
                    text: page.state.error || (page.state.busy ? "Loading Megasweep data…" : "Open a file to begin.")
                    color: page.state.error ? page.colors.error : page.colors.muted
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                }
            }

        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: 8
            spacing: 3
            Text { objectName: "megasweep_source"; Layout.fillWidth: true; text: page.state.sourceFile || page.state.sourcePath || ""; color: page.colors.text; elide: Text.ElideMiddle; font.pixelSize: 12 }
            Text { objectName: "megasweep_summary"; Layout.fillWidth: true; text: page.state.summary || ""; visible: text.length > 0; color: page.colors.text; wrapMode: Text.Wrap; font.pixelSize: 12 }
            Text { objectName: "megasweep_preview_notice"; Layout.fillWidth: true; visible: page.state.reflectionPreviewVisible === true; text: "Reference spectra preview. Select Analysis map to export the analysis."; color: page.colors.muted; wrapMode: Text.Wrap; font.pixelSize: 11 }
            Text { objectName: "megasweep_status"; Layout.fillWidth: true; text: page.state.error || page.state.status || ""; color: page.state.error ? page.colors.error : page.colors.muted; wrapMode: Text.Wrap; font.pixelSize: 11; Accessible.name: text }
            Basic.ProgressBar { Layout.fillWidth: true; visible: page.state.busy === true; indeterminate: true }
        }
    }
}
