pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: panel
    objectName: "megasweep_plot_limits"
    required property var bridge
    required property var colors
    required property bool active
    property var rows: []
    readonly property bool compactHeaders: width < (xEditor.singleLineWidth + yEditor.singleLineWidth
        + axesFlow.spacing + 2 * axesBar.padding
        + (colorBar.visible ? colorEditor.singleLineWidth + 2 * colorBar.padding + bars.spacing : 0))
    readonly property int fieldWidth: width < 260 ? 59 : 69
    spacing: 3
    visible: active && rows.length > 0
    Accessible.role: Accessible.Pane
    Accessible.name: "Plot limits"

    component LimitValue: Basic.TextField {
        id: valueInput
        property string fullValueText: ""
        property bool edited: false
        TextMetrics { id: valueMetrics; font: valueInput.font; text: valueInput.text }
        Layout.preferredWidth: Math.max(panel.fieldWidth, Math.min(panel.width - 40,
                                        Math.ceil(valueMetrics.advanceWidth) + leftPadding + rightPadding + 3))
        Layout.minimumWidth: Layout.preferredWidth
        implicitHeight: 24
        leftPadding: 4
        rightPadding: 4
        selectByMouse: true
        hoverEnabled: true
        Basic.ToolTip.visible: hovered && fullValueText.length > 0
        Basic.ToolTip.text: fullValueText
        Basic.ToolTip.delay: 650
        onTextChanged: if (!activeFocus) cursorPosition = 0
        onActiveFocusChanged: if (!activeFocus) cursorPosition = 0
    }

    function refresh() {
        const updated = active && bridge ? bridge.mapLimits() : []
        if (JSON.stringify(updated) !== JSON.stringify(rows)) {
            rows = updated
        }
    }
    onActiveChanged: refresh()
    Component.onCompleted: refresh()
    Connections {
        target: panel.bridge
        function onPlotLimitsChanged() { panel.refresh() }
    }

    component LimitGroup: ColumnLayout {
        id: row
        required property var info
        property string inputError: ""
        readonly property string bounds: JSON.stringify([info.key, info.label, info.low, info.high, info.automatic])
        readonly property real fieldPairWidth: lowerField.implicitWidth + upperField.implicitWidth + fields.spacing
        readonly property real singleLineWidth: Math.ceil(Math.max(fieldPairWidth,
            titleMetrics.advanceWidth + header.spacing + headerActions.implicitWidth))
        readonly property real preferredWidth: panel.compactHeaders
            ? Math.ceil(Math.max(fieldPairWidth, titleMetrics.advanceWidth, headerActions.implicitWidth)) : singleLineWidth
        objectName: "megasweep_limits_group_" + (info.key || "")
        spacing: 2
        function formatted(value) {
            const autoColor = info.key === "color_auto" && info.automatic
            const rounded = Number(Number(value).toPrecision(autoColor ? 4 : 6))
            const magnitude = Math.abs(rounded)
            if (autoColor && magnitude > 0 && (magnitude < 0.001 || magnitude >= 10000))
                return rounded.toExponential()
            return String(rounded)
        }
        function loadBounds() {
            inputError = ""
            minimum.text = formatted(info.low)
            maximum.text = formatted(info.high)
            minimum.edited = false
            maximum.edited = false
            automatic.checked = !!info.automatic
        }
        function limits() {
            return {key: info.key, automatic: automatic.checked,
                    low: !minimum.edited && minimum.text === formatted(info.low) ? info.low : minimum.text,
                    high: !maximum.edited && maximum.text === formatted(info.high) ? info.high : maximum.text}
        }
        function commit(force) {
            if (!panel.visible || !panel.enabled || !info.key) return
            if (!force && !minimum.edited && !maximum.edited) return
            inputError = panel.bridge.applyMapLimits([limits()])
            if (!inputError) {
                panel.refresh()
                loadBounds()
            }
        }
        onBoundsChanged: if (minimum && maximum && automatic) loadBounds()
        Component.onCompleted: loadBounds()

        Flow {
            id: header
            Layout.fillWidth: true
            spacing: 4
            Basic.Label {
                id: title
                objectName: "megasweep_limit_title_" + (row.info.key || "")
                TextMetrics { id: titleMetrics; font: title.font; text: title.text }
                width: Math.min(row.width, Math.ceil(titleMetrics.advanceWidth))
                height: Math.max(22, contentHeight)
                verticalAlignment: Text.AlignVCenter
                wrapMode: Text.WrapAnywhere
                text: (row.info.axis || "") + (row.info.label ? " · " + row.info.label : "")
                color: panel.colors.text
                font.bold: true
            }
            RowLayout {
                id: headerActions
                spacing: 4
                AppComboBox {
                    id: paletteChoice
                    objectName: "megasweep_colormap_" + (row.info.key || "")
                    visible: row.info.key === "color_auto"
                    colors: panel.colors
                    model: row.info.colormaps || []
                    currentIndex: model.indexOf(row.info.cmap)
                    implicitWidth: Math.ceil(paletteMetrics.advanceWidth) + leftPadding + rightPadding + 14
                    implicitHeight: 24
                    TextMetrics { id: paletteMetrics; font: paletteChoice.font; text: "RdBu_r" }
                    Accessible.name: "Map colormap"
                    onActivated: index => panel.bridge.setParameter("cmap", model[index])
                }
                Basic.CheckBox {
                    id: automatic
                    objectName: "megasweep_limit_auto_" + row.info.key
                    text: "Auto"
                    spacing: 3
                    leftPadding: 0
                    rightPadding: 0
                    implicitHeight: 22
                    Accessible.name: "Automatic " + row.info.axis + " limits"
                    onClicked: row.commit(true)
                }
                AppButton {
                    id: reverseDirection
                    objectName: "megasweep_reverse_" + row.info.key
                    colors: panel.colors
                    compact: true
                    visible: row.info.key === "x_auto" || row.info.key === "y_auto"
                    text: row.info.reversed ? "High → low" : "Low → high"
                    implicitHeight: 22
                    description: "Reverse " + row.info.axis + " direction"
                    onClicked: panel.bridge.setParameter(row.info.key === "x_auto" ? "x_reverse" : "y_reverse", !row.info.reversed)
                }
            }
        }
        Flow {
            id: fields
            Layout.fillWidth: true
            spacing: 4
            RowLayout {
                id: lowerField
                spacing: 4
                Basic.Label { text: "Min"; color: panel.colors.muted }
                LimitValue {
                    id: minimum
                    objectName: "megasweep_limit_min_" + row.info.key
                    fullValueText: !edited && text === row.formatted(row.info.low) ? String(row.info.low) : text
                    Accessible.name: row.info.axis + " minimum " + row.info.label
                    onTextEdited: { edited = true; automatic.checked = false }
                    onEditingFinished: row.commit(false)
                    Keys.onEscapePressed: row.loadBounds()
                }
            }
            RowLayout {
                id: upperField
                spacing: 4
                Basic.Label { text: "Max"; color: panel.colors.muted }
                LimitValue {
                    id: maximum
                    objectName: "megasweep_limit_max_" + row.info.key
                    fullValueText: !edited && text === row.formatted(row.info.high) ? String(row.info.high) : text
                    Accessible.name: row.info.axis + " maximum " + row.info.label
                    onTextEdited: { edited = true; automatic.checked = false }
                    onEditingFinished: row.commit(false)
                    Keys.onEscapePressed: row.loadBounds()
                }
            }
        }
        Basic.Label {
            Layout.fillWidth: true
            visible: !!row.inputError
            wrapMode: Text.Wrap
            text: row.inputError
            color: panel.colors.error
        }
    }

    component LimitBar: Basic.Pane {
        padding: 6
        background: Rectangle {
            color: panel.colors.surface
            border.color: panel.colors.border
            radius: 4
        }
    }

    Flow {
        id: bars
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        spacing: 8

        LimitBar {
            id: axesBar
            objectName: "megasweep_coordinate_limits_bar"
            width: Math.min(bars.width, xEditor.preferredWidth + yEditor.preferredWidth
                            + axesFlow.spacing + 2 * padding)
            contentItem: Flow {
                id: axesFlow
                spacing: 12
                LimitGroup {
                    id: xEditor
                    info: panel.rows[0] || ({})
                    width: Math.min(axesBar.availableWidth, preferredWidth)
                }
                LimitGroup {
                    id: yEditor
                    info: panel.rows[1] || ({})
                    width: Math.min(axesBar.availableWidth, preferredWidth)
                }
            }
        }

        LimitBar {
            id: colorBar
            objectName: "megasweep_color_limits_bar"
            visible: panel.rows.length > 2
            width: Math.min(bars.width, colorEditor.preferredWidth + 2 * padding)
            contentItem: LimitGroup {
                id: colorEditor
                info: panel.rows[2] || ({})
            }
        }
    }
}
