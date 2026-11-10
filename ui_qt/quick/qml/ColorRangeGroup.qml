pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: group
    required property var controller
    required property var colors
    required property var descriptor
    readonly property string groupId: descriptor.id || ""
    readonly property bool dragging: range.first.pressed || range.second.pressed
    property bool synchronizing: false
    property var dragMapping: ({})
    property int contextVersion: 0
    property int dragContext: -1
    readonly property var liveState: controller ? controller.preview : ({})
    readonly property var liveValues: liveState.contextVersion === contextVersion ? (liveState.groups || {})[groupId] || ({}) : ({})
    spacing: 2
    objectName: "color_range_group_" + groupId
    enabled: descriptor.enabled !== false

    function valueAt(position) {
        let m = dragging ? dragMapping : descriptor
        if (position <= 0) return m.domainLow
        if (position >= 1) return m.domainHigh
        let t = interpolate(position, m.mappingPositions || [0, 1], m.mappingValues || [m.tLow, m.tHigh])
        if (m.kind === "log") return Math.exp(t)
        if (m.kind === "asinh") return m.center + m.softness * Math.sinh(t)
        return t
    }
    function interpolate(value, input, output) {
        let i = 1
        while (i < input.length - 1 && value > input[i]) ++i
        let fraction = Math.max(0, Math.min(1, (value - input[i - 1]) / (input[i] - input[i - 1])))
        return output[i - 1] + fraction * (output[i] - output[i - 1])
    }
    function positionAt(value) {
        let m = dragging ? dragMapping : descriptor
        let t = m.kind === "log" ? Math.log(value)
              : m.kind === "asinh" ? Math.asinh((value - m.center) / m.softness) : value
        return interpolate(t, m.mappingValues || [m.tLow, m.tHigh], m.mappingPositions || [0, 1])
    }
    function sync() {
        if (dragging) return
        synchronizing = true
        range.setValues(descriptor.first === undefined ? 0 : descriptor.first,
                        descriptor.second === undefined ? 1 : descriptor.second)
        synchronizing = false
    }
    function submit(commitChange) {
        if (synchronizing || !controller || !groupId) return
        if (dragContext >= 0 && dragContext !== contextVersion) return
        controller.moveRange(groupId, range.first.value, range.second.value, commitChange)
    }
    function moved(endpoint) {
        if (descriptor.fixedLow) range.first.value = descriptor.first
        if (descriptor.fixedHigh) range.second.value = descriptor.second
        if (descriptor.symmetric && !descriptor.fixedLow && !descriptor.fixedHigh) {
            let amplitude = Math.abs(valueAt(endpoint === "low" ? range.first.value : range.second.value))
            range.setValues(positionAt(-amplitude), positionAt(amplitude))
        }
        submit(!dragging)
    }
    onDescriptorChanged: sync()
    onDraggingChanged: {
        if (dragging) { dragMapping = descriptor; dragContext = contextVersion }
        else { submit(true); dragContext = -1; sync() }
    }
    Component.onCompleted: sync()

    RowLayout {
        Layout.fillWidth: true
        spacing: 2
        Text {
            Layout.fillWidth: true
            Layout.minimumWidth: 28
            text: group.descriptor.title || "Colors"
            color: group.colors.text
            font.weight: Font.DemiBold
            elide: Text.ElideRight
        }
        AppButton {
            objectName: "color_overall_" + group.groupId
            colors: group.colors; compact: true; text: "Overall"
            visible: group.descriptor.canRecommend === true
            description: "Recommend overall colors for " + group.descriptor.title
            onClicked: group.controller.recommend(group.groupId, "overall")
        }
        AppButton {
            objectName: "color_weak_" + group.groupId
            colors: group.colors; compact: true; text: "Weak features"
            visible: group.descriptor.canRecommend === true
            description: "Recommend weak feature colors for " + group.descriptor.title
            onClicked: group.controller.recommend(group.groupId, "weak")
        }
        AppButton {
            objectName: "color_auto_" + group.groupId
            colors: group.colors; compact: true; text: "Auto"
            description: "Automatic colors for " + group.descriptor.title + "; retain fixed endpoints"
            onClicked: group.controller.autoRange(group.groupId)
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text {
            objectName: "color_min_readout_" + group.groupId
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            text: "Min " + (group.liveValues.lowText || group.descriptor.lowText || "")
                  + (group.descriptor.fixedLow ? " · Fixed" : "")
            color: group.colors.text
            font.pointSize: 9
            elide: Text.ElideRight
            Accessible.name: group.descriptor.title + " " + text
        }
        Text {
            objectName: "color_max_readout_" + group.groupId
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            text: "Max " + (group.liveValues.highText || group.descriptor.highText || "")
                  + (group.descriptor.fixedHigh ? " · Fixed" : "")
            color: group.colors.text
            font.pointSize: 9
            elide: Text.ElideRight
            horizontalAlignment: Text.AlignRight
            Accessible.name: group.descriptor.title + " " + text
        }
    }
    Basic.RangeSlider {
        id: range
        objectName: "color_slider_" + group.groupId
        Layout.fillWidth: true
        from: 0; to: 1; stepSize: 0.001
        implicitHeight: 26
        leftPadding: 8; rightPadding: 8
        first.onMoved: group.moved("low")
        second.onMoved: group.moved("high")
        first.handle: Rectangle {
            x: range.leftPadding + range.first.visualPosition * (range.availableWidth - width)
            y: range.topPadding + range.availableHeight / 2 - height / 2
            implicitWidth: 16; implicitHeight: 22; radius: 4
            color: group.descriptor.fixedLow ? group.colors.inset : group.colors.control
            border.width: range.first.pressed || range.activeFocus ? 2 : 1
            border.color: range.activeFocus ? group.colors.focus : group.colors.accent
            Text { anchors.centerIn: parent; text: "L"; font.pixelSize: 10; color: group.colors.text }
        }
        second.handle: Rectangle {
            x: range.leftPadding + range.second.visualPosition * (range.availableWidth - width)
            y: range.topPadding + range.availableHeight / 2 - height / 2
            implicitWidth: 16; implicitHeight: 22; radius: 4
            color: group.descriptor.fixedHigh ? group.colors.inset : group.colors.control
            border.width: range.second.pressed || range.activeFocus ? 2 : 1
            border.color: range.activeFocus ? group.colors.focus : group.colors.accent
            Text { anchors.centerIn: parent; text: "H"; font.pixelSize: 10; color: group.colors.text }
        }
        background: Rectangle {
            x: range.leftPadding + 8; y: range.topPadding + range.availableHeight / 2 - height / 2
            width: Math.max(0, range.availableWidth - 16); height: 4; radius: 2
            color: group.colors.border
            Rectangle {
                x: range.first.position * parent.width
                width: (range.second.position - range.first.position) * parent.width
                height: parent.height; radius: 2; color: group.colors.accent
            }
        }
        Accessible.name: group.descriptor.title + " color range"
        Accessible.description: "Two handles: minimum and maximum. Nonlinear scale; full data range remains available."
        Basic.ToolTip.visible: hovered && !group.dragging
        Basic.ToolTip.text: "Full range " + group.descriptor.domainLowText + " to " + group.descriptor.domainHighText
                            + " · nonlinear spacing"
    }
    RowLayout {
        Layout.fillWidth: true
        Text { text: group.descriptor.domainLowText || ""; color: group.colors.muted; font.pointSize: 8 }
        Item { Layout.fillWidth: true }
        Text { text: group.descriptor.domainMidText || ""; color: group.colors.muted; font.pointSize: 8 }
        Item { Layout.fillWidth: true }
        Text { text: group.descriptor.domainHighText || ""; color: group.colors.muted; font.pointSize: 8 }
    }
    Text {
        Layout.fillWidth: true
        text: group.descriptor.status || ""
        visible: text.length > 0
        elide: Text.ElideRight
        color: group.colors.muted; font.pointSize: 8
        Basic.ToolTip.visible: statusHover.hovered && text.length > 0
        Basic.ToolTip.text: text
        HoverHandler { id: statusHover }
    }
}
