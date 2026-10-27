pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

ColumnLayout {
    id: field
    required property var descriptor
    required property var bridge
    required property var colors
    property string caption: descriptor.label || descriptor.key || ""
    property bool showLabel: true
    spacing: 3
    visible: descriptor.visible !== false

    Basic.Label {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        visible: field.showLabel && field.descriptor.type !== "flag" && field.descriptor.type !== "action"
                 && (field.descriptor.type !== "summary" || !!field.descriptor.label)
        text: field.caption
        color: field.colors.muted
        wrapMode: Text.WordWrap
    }
    Loader {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        active: field.descriptor.type !== undefined
        sourceComponent: field.descriptor.type === "number" ? numberEditor
                       : field.descriptor.type === "choice" ? choiceEditor
                       : field.descriptor.type === "flag" ? flagEditor
                       : field.descriptor.type === "action" ? actionEditor
                       : field.descriptor.type === "summary" ? summaryEditor : textEditor
        Component {
            id: numberEditor
            AppNumberField { descriptor: field.descriptor; bridge: field.bridge; colors: field.colors }
        }
        Component {
            id: choiceEditor
            AppComboBox {
                colors: field.colors
                id: choice
                readonly property string optionsSignature: JSON.stringify(field.descriptor.options || [])
                objectName: "app_parameter_" + field.descriptor.key
                implicitHeight: Math.max(30, (contentItem ? contentItem.implicitHeight : 0) + 10)
                enabled: field.descriptor.enabled !== false && field.descriptor.readOnly !== true
                model: JSON.parse(optionsSignature).map(function(option) {
                    return typeof option === "object" ? option.label : String(option)
                })
                currentIndex: {
                    let options = field.descriptor.options || []
                    for (let index = 0; index < options.length; ++index) {
                        let value = typeof options[index] === "object" ? options[index].value : options[index]
                        if (value === field.descriptor.value) return index
                    }
                    return -1
                }
                Accessible.name: field.descriptor.label || field.descriptor.key || ""
                Accessible.description: field.descriptor.description || ""
                hoverEnabled: true
                Basic.ToolTip.visible: choice.hovered && !!field.descriptor.description
                Basic.ToolTip.text: field.descriptor.description || ""
                Basic.ToolTip.delay: 700
                onActivated: function(index) {
                    let option = field.descriptor.options[index]
                    field.bridge.setParameter(field.descriptor.key, typeof option === "object" ? option.value : option)
                }
            }
        }
        Component {
            id: flagEditor
            Basic.CheckBox {
                id: flag
                objectName: "app_parameter_" + field.descriptor.key
                text: field.descriptor.label || field.descriptor.key || ""
                checked: field.descriptor.value === true
                enabled: field.descriptor.enabled !== false && field.descriptor.readOnly !== true
                Accessible.name: text
                Accessible.description: field.descriptor.description || ""
                onToggled: field.bridge.setParameter(field.descriptor.key, checked)
                contentItem: Text {
                    text: flag.text
                    color: flag.enabled ? field.colors.text : field.colors.disabled
                    leftPadding: flag.indicator.width + flag.spacing
                    verticalAlignment: Text.AlignVCenter
                    wrapMode: Text.WordWrap
                    font: flag.font
                }
            }
        }
        Component {
            id: actionEditor
            AppButton {
                id: action
                objectName: "app_parameter_" + (field.descriptor.id || field.descriptor.key)
                colors: field.colors
                text: field.descriptor.label || field.descriptor.key || ""
                description: field.descriptor.description || text
                implicitHeight: Math.max(30, (contentItem ? contentItem.implicitHeight : 0) + 10)
                enabled: field.descriptor.enabled !== false
                primary: field.descriptor.primary === true
                onClicked: field.bridge.invoke(field.descriptor.id || field.descriptor.key)
                contentItem: Text {
                    text: action.text
                    color: action.enabled ? (action.primary ? field.colors.selectedText : field.colors.text) : field.colors.disabled
                    font: action.font
                    wrapMode: Text.WordWrap
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }
        }
        Component {
            id: summaryEditor
            Basic.Label {
                objectName: "app_parameter_" + field.descriptor.key
                text: field.descriptor.value === undefined ? "" : String(field.descriptor.value)
                wrapMode: Text.Wrap
                color: field.colors.muted
                Accessible.name: field.descriptor.label || text
                Accessible.description: text
            }
        }
        Component {
            id: textEditor
            Basic.TextField {
                id: textField
                FontMetrics { id: textMetrics; font: textField.font }
                property bool dirty: false
                readonly property string canonicalText: field.descriptor.value === undefined ? "" : String(field.descriptor.value)
                objectName: "app_parameter_" + field.descriptor.key
                implicitHeight: Math.max(30, textMetrics.height + 10)
                enabled: field.descriptor.enabled !== false
                readOnly: field.descriptor.readOnly === true
                selectByMouse: true
                Accessible.name: field.descriptor.label || field.descriptor.key || ""
                Accessible.description: field.descriptor.description || ""
                function refresh() { if (!dirty) text = canonicalText }
                onCanonicalTextChanged: refresh()
                Component.onCompleted: refresh()
                onTextEdited: dirty = true
                onEditingFinished: {
                    if (dirty && !readOnly) {
                        dirty = false
                        if (text !== canonicalText) field.bridge.setParameter(field.descriptor.key, text)
                    }
                }
                Keys.onEscapePressed: { dirty = false; refresh() }
            }
        }
    }
}
