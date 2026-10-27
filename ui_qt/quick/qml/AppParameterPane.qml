pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Control {
    id: pane
    required property var bridge
    required property var colors
    property var groups: []
    property string message: ""
    property var groupSchema: []
    property var descriptors: ({})
    property var groupDescriptors: ({})
    property var expandedGroups: ({})
    property string schemaSignature: ""
    readonly property var pinnedSchema: groupSchema.filter(function(group) { return group.pinned })
    readonly property var scrollingSchema: groupSchema.filter(function(group) { return !group.pinned })
    implicitWidth: 320
    Accessible.role: Accessible.Pane
    Accessible.name: "Workflow controls"
    background: Rectangle { color: pane.colors.surface }

    // Value/label/status snapshots retain editor instances and unfinished drafts.
    function synchronizeSchema() {
        let fields = {}, headings = {}
        let schema = groups.map(function(group) {
            let groupKey = group.key || group.title || "controls"
            headings[groupKey] = group
            let rows = [], values = group.fields || []
            for (let index = 0; index < values.length; ++index) {
                let field = values[index], key = field.key || field.id
                fields[key] = field
                let next = values[index + 1]
                if (field.type === "number" && field.pair && next && next.type === "number" && next.pair === field.pair) {
                    let nextKey = next.key || next.id
                    fields[nextKey] = next
                    rows.push({key: "pair:" + field.pair, type: "pair", first: key, second: nextKey})
                    ++index
                } else if (field.type === "action") {
                    let keys = [key]
                    while (values[index + 1] && values[index + 1].type === "action") {
                        let action = values[++index], actionKey = action.key || action.id
                        fields[actionKey] = action
                        keys.push(actionKey)
                    }
                    rows.push({key: "actions:" + key, type: "actions", keys: keys})
                } else rows.push({key: key, type: field.type})
            }
            return {key: groupKey, rows: rows,
                    pinned: group.pinned === true || groupKey.toLowerCase() === "data"}
        })
        descriptors = fields
        groupDescriptors = headings
        let signature = JSON.stringify(schema)
        if (signature !== schemaSignature) {
            schemaSignature = signature
            groupSchema = schema
        }
    }
    function isExpanded(key) {
        let descriptor = groupDescriptors[key] || {}
        if (expandedGroups[key] !== undefined) return expandedGroups[key]
        return descriptor.expanded !== undefined ? descriptor.expanded : descriptor.advanced !== true
    }
    function toggleGroup(key) {
        let next = Object.assign({}, expandedGroups)
        next[key] = !isExpanded(key)
        expandedGroups = next
    }
    onGroupsChanged: synchronizeSchema()
    Component.onCompleted: synchronizeSchema()

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        ColumnLayout {
            Layout.fillWidth: true
            Layout.leftMargin: pane.width < 260 ? 10 : 12
            Layout.rightMargin: pane.width < 260 ? 10 : 12
            Layout.topMargin: visible ? 8 : 0
            Layout.bottomMargin: visible ? 4 : 0
            visible: pane.pinnedSchema.length > 0
            spacing: 8
            Repeater {
                model: pane.pinnedSchema
                delegate: AppParameterGroup {
                    required property var modelData
                    Layout.fillWidth: true
                    schema: modelData
                    descriptor: pane.groupDescriptors[modelData.key] || ({})
                    descriptors: pane.descriptors
                    bridge: pane.bridge
                    colors: pane.colors
                    expanded: true
                }
            }
        }
        Basic.ScrollView {
            id: scroll
            objectName: "app_parameter_scroll"
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: availableWidth
            clip: true
            padding: width < 260 ? 10 : 12
            Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
            ColumnLayout {
                width: scroll.availableWidth
                spacing: 14
                Basic.Label {
                    Layout.fillWidth: true
                    visible: pane.message.length > 0
                    text: pane.message
                    color: pane.colors.muted
                    wrapMode: Text.WordWrap
                    Accessible.name: text
                }
                Repeater {
                    model: pane.scrollingSchema
                    delegate: AppParameterGroup {
                        required property var modelData
                        Layout.fillWidth: true
                        schema: modelData
                        descriptor: pane.groupDescriptors[modelData.key] || ({})
                        descriptors: pane.descriptors
                        bridge: pane.bridge
                        colors: pane.colors
                        expanded: pane.isExpanded(modelData.key)
                        onToggleRequested: pane.toggleGroup(modelData.key)
                    }
                }
            }
        }
    }
}
