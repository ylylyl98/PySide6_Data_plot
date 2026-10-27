pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: group
    required property var schema
    required property var descriptor
    required property var descriptors
    required property var bridge
    required property var colors
    property bool expanded: true
    readonly property bool collapsible: descriptor.advanced === true || descriptor.collapsible === true
    signal toggleRequested()
    spacing: 7

    Basic.ToolButton {
        id: heading
        objectName: "app_parameter_group_" + group.schema.key
        Layout.fillWidth: true
        implicitHeight: Math.max(28, (contentItem ? contentItem.implicitHeight : 0) + 8)
        padding: 0
        focusPolicy: group.collapsible ? Qt.StrongFocus : Qt.NoFocus
        Accessible.role: group.collapsible ? Accessible.Button : Accessible.Heading
        Accessible.name: group.descriptor.title || group.schema.key
        Accessible.description: group.collapsible ? (group.expanded ? "Collapse advanced settings" : "Expand advanced settings")
                                                 : group.descriptor.description || ""
        onClicked: if (group.collapsible) group.toggleRequested()
        contentItem: RowLayout {
            spacing: 6
            Basic.Label {
                Layout.fillWidth: true
                text: group.descriptor.title || group.schema.key
                font.weight: Font.DemiBold
                color: group.colors.text
                wrapMode: Text.WordWrap
            }
            Item {
                visible: group.collapsible
                implicitWidth: 12
                implicitHeight: 12
                Rectangle { anchors.centerIn: parent; width: 10; height: 1.5; color: group.colors.muted }
                Rectangle { anchors.centerIn: parent; width: 1.5; height: 10; color: group.colors.muted; visible: !group.expanded }
            }
        }
        background: Rectangle {
            radius: 4
            color: group.collapsible && heading.hovered ? group.colors.hover : "transparent"
            border.width: heading.visualFocus ? 2 : 0
            border.color: group.colors.focus
        }
    }
    ColumnLayout {
        objectName: "app_parameter_group_body_" + group.schema.key
        Layout.fillWidth: true
        visible: !group.collapsible || group.expanded
        spacing: 9
        Basic.Label {
            Layout.fillWidth: true
            visible: text.length > 0
            text: group.descriptor.description || ""
            color: group.colors.muted
            wrapMode: Text.WordWrap
        }
        Repeater {
            model: group.schema.rows
            delegate: Loader {
                id: row
                required property var modelData
                readonly property var descriptor: group.descriptors[modelData.key] || ({})
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                visible: modelData.type === "pair" || modelData.type === "actions" || descriptor.visible !== false
                sourceComponent: modelData.type === "pair" ? pairEditor : modelData.type === "actions" ? actionsEditor : singleEditor
                Component {
                    id: pairEditor
                    AppRangeField {
                        firstDescriptor: group.descriptors[row.modelData.first] || ({})
                        secondDescriptor: group.descriptors[row.modelData.second] || ({})
                        bridge: group.bridge
                        colors: group.colors
                    }
                }
                Component {
                    id: singleEditor
                    AppParameterField { descriptor: row.descriptor; bridge: group.bridge; colors: group.colors }
                }
                Component {
                    id: actionsEditor
                    Flow {
                        id: flow
                        spacing: 5
                        Repeater {
                            model: row.modelData.keys
                            delegate: AppButton {
                                id: action
                                required property string modelData
                                readonly property var descriptor: group.descriptors[modelData] || ({})
                                objectName: "app_parameter_" + (descriptor.id || descriptor.key)
                                colors: group.colors
                                text: descriptor.label || descriptor.key
                                description: descriptor.description || text
                                width: Math.min(implicitWidth, flow.width)
                                implicitHeight: Math.max(30, (contentItem ? contentItem.implicitHeight : 0) + 10)
                                enabled: descriptor.enabled !== false
                                visible: descriptor.visible !== false
                                primary: descriptor.primary === true
                                onClicked: group.bridge.invoke(descriptor.id || descriptor.key)
                                contentItem: Text {
                                    text: action.text
                                    font: action.font
                                    color: action.enabled ? (action.primary ? group.colors.selectedText : group.colors.text) : group.colors.disabled
                                    wrapMode: Text.WordWrap
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: group.colors.divider }
}
