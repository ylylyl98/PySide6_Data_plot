pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window

Rectangle {
    id: root
    required property var bridge
    property var colors: ({
        "surface": "#f3f5f8", "control": "#ffffff", "text": "#20242b",
        "muted": "#606978", "border": "#c6cbd3", "accent": "#0067c0",
        "hover": "#edf1f6", "pressed": "#e0e7ef", "error": "#b42318",
        "disabled": "#939aa5", "selectedText": "#ffffff",
        "inset": "#e9edf2", "divider": "#dce1e8", "borderHover": "#aeb7c4",
        "borderDisabled": "#dce1e8", "controlDisabled": "#edf0f4", "focus": "#20242b",
        "accentText": "#0067c0", "selected": "#e6f0fa", "selectedBorder": "#b4d6fa",
        "errorSurface": "#fdf1f0"
    })
    readonly property var displayState: bridge.snapshot
    implicitWidth: 360
    implicitHeight: 680
    color: colors.surface
    objectName: "drr_quick_display"
    Accessible.role: Accessible.Pane
    Accessible.name: "DRR display settings"

    // Cross the Python/QVariant boundary once per snapshot. Every editor
    // shares this JavaScript value instead of converting the whole map for
    // each descriptor binding during held-arrow updates.
    QtObject {
        id: viewBridge
        readonly property var snapshot: root.displayState
        readonly property string error: root.bridge.error
        signal deactivating()

        function setNumber(key, text) { return root.bridge.setNumber(key, text) }
        function stepNumber(key, direction, draft) {
            return draft === undefined ? root.bridge.stepNumber(key, direction)
                                       : root.bridge.stepNumber(key, direction, draft)
        }
        function setChoice(key, index) { return root.bridge.setChoice(key, index) }
        function setFlag(key, checked) { return root.bridge.setFlag(key, checked) }
        function invoke(action) { return root.bridge.invoke(action) }
        function setEditing(key, editing) { root.bridge.setEditing(key, editing) }
    }

    Connections {
        target: root.bridge
        function onDeactivating() { viewBridge.deactivating() }
    }

    function ensureFocusVisible() {
        let window = root.Window.window
        let item = window ? window.activeFocusItem : null
        let ancestor = item
        while (ancestor && ancestor !== form)
            ancestor = ancestor.parent
        if (!item || ancestor !== form)
            return
        let position = item.mapToItem(form, 0, 0)
        let flick = scroll.contentItem as Flickable
        let lowerEdge = position.y + item.height + 8
        if (position.y < flick.contentY + 8)
            flick.contentY = Math.max(0, position.y - 8)
        else if (lowerEdge > flick.contentY + flick.height)
            flick.contentY = Math.min(Math.max(0, flick.contentHeight - flick.height), lowerEdge - flick.height)
    }

    Connections {
        target: root.Window.window
        function onActiveFocusItemChanged() { Qt.callLater(root.ensureFocusVisible) }
    }

    component SectionTitle: ColumnLayout {
        id: sectionTitle
        property string text
        property string caption: ""
        Layout.fillWidth: true
        spacing: 3
        Text {
            Layout.fillWidth: true
            text: sectionTitle.text
            color: root.colors.text
            font.pointSize: 10
            font.weight: Font.DemiBold
        }
        Text {
            Layout.fillWidth: true
            text: sectionTitle.caption
            visible: text.length > 0
            color: root.colors.muted
            font.pointSize: 8
            wrapMode: Text.Wrap
        }
    }

    component Divider: Rectangle {
        Layout.fillWidth: true
        Layout.topMargin: 5
        Layout.bottomMargin: 4
        implicitHeight: 1
        color: root.colors.divider
    }

    component ColorTab: Basic.TabButton {
        id: tab
        implicitHeight: 30
        font.pointSize: 9
        hoverEnabled: true
        focusPolicy: Qt.StrongFocus
        Accessible.name: "Edit " + text + " colors"
        Accessible.description: "Choose the color controls to edit. The displayed plot stays selected."
        Basic.ToolTip.visible: hovered
        Basic.ToolTip.text: "Edit " + text + " colors; plot selection stays unchanged."
        Basic.ToolTip.delay: 650
        contentItem: Text {
            text: tab.text
            font.pointSize: tab.font.pointSize
            font.weight: tab.checked ? Font.DemiBold : Font.Normal
            color: !tab.enabled ? root.colors.disabled
                                : tab.checked ? root.colors.accentText : root.colors.muted
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            color: tab.down ? root.colors.pressed : tab.checked ? root.colors.control
                                : tab.hovered ? root.colors.hover : "transparent"
            radius: 5
            border.width: tab.visualFocus ? 2 : tab.checked ? 1 : 0
            border.color: tab.visualFocus ? root.colors.focus : root.colors.border
        }
    }

    Basic.ScrollView {
        id: scroll
        objectName: "drr_display_scroll"
        anchors.fill: parent
        clip: true
        padding: 16
        rightPadding: 16
        contentWidth: availableWidth
        Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
        Basic.ScrollBar.vertical: Basic.ScrollBar {
            id: verticalBar
            objectName: "drr_display_scrollbar_vertical"
            parent: scroll
            anchors.top: scroll.top
            anchors.bottom: scroll.bottom
            anchors.right: scroll.right
            anchors.topMargin: 2
            anchors.bottomMargin: 2
            anchors.rightMargin: 1
            policy: Basic.ScrollBar.AsNeeded
            width: 8
            contentItem: Rectangle {
                implicitWidth: 5
                implicitHeight: 30
                radius: 3
                color: verticalBar.pressed ? root.colors.accent : root.colors.muted
                opacity: verticalBar.hovered || verticalBar.pressed ? 0.7 : 0.4
            }
            background: null
        }

        ColumnLayout {
            id: form
            width: scroll.availableWidth
            spacing: 10

            SectionTitle {
                text: "Axes"
                caption: "Fix keeps a limit when Auto adjusts the range."
            }
            RangeEditor {
                Layout.fillWidth: true
                bridge: viewBridge
                colors: root.colors
                title: "X"
                minimumKey: "axis.xmin"
                maximumKey: "axis.xmax"
                autoAction: "auto.x"
            }
            RangeEditor {
                Layout.fillWidth: true
                bridge: viewBridge
                colors: root.colors
                title: "Y"
                minimumKey: "axis.ymin"
                maximumKey: "axis.ymax"
                autoAction: "auto.y"
            }
            ColumnLayout {
                Layout.fillWidth: true
                Layout.topMargin: 2
                spacing: 5
                Text { text: "Left Y-axis"; color: root.colors.muted; font.pointSize: 8 }
                ChoiceEditor {
                    Layout.fillWidth: true
                    bridge: viewBridge
                    colors: root.colors
                    fieldKey: "yaxis.left"
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                visible: root.displayState.advancedYAxis === true
                spacing: 6
                Text {
                    Layout.fillWidth: true
                    text: (root.displayState.texts.yaxisEquation || {}).text || "Y = a × TG + b × BG + c"
                    color: root.colors.muted
                    font.pointSize: 8
                    wrapMode: Text.Wrap
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        spacing: 4
                        Text { text: "a · TG"; color: root.colors.muted; font.pointSize: 8 }
                        NumberEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "yaxis.a" }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        spacing: 4
                        Text { text: "b · BG"; color: root.colors.muted; font.pointSize: 8 }
                        NumberEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "yaxis.b" }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        spacing: 4
                        Text { text: "c · offset"; color: root.colors.muted; font.pointSize: 8 }
                        NumberEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "yaxis.c" }
                    }
                }
                Text {
                    Layout.fillWidth: true
                    text: (root.displayState.texts.yaxisLabel || {}).text || ""
                    visible: text.length > 0
                    color: root.colors.muted
                    font.pointSize: 8
                    wrapMode: Text.Wrap
                }
            }

            Divider { }
            SectionTitle {
                text: "Color scale"
                caption: "Choose the product whose colors you want to edit."
            }
            Basic.TabBar {
                id: colorTabs
                objectName: "drr_color_tabs"
                Layout.fillWidth: true
                padding: 3
                spacing: 3
                currentIndex: root.displayState.colorProduct || 0
                Accessible.name: "Edit colors for product"
                background: Rectangle {
                    color: root.colors.inset
                    border.color: root.colors.divider
                    radius: 7
                }
                onCurrentIndexChanged: {
                    if (currentIndex >= 0 && currentIndex !== root.displayState.colorProduct)
                        root.bridge.setChoice("colorProduct", currentIndex)
                }
                ColorTab { objectName: "drr_color_tab_raw"; text: "ΔR/R" }
                ColorTab { objectName: "drr_color_tab_second"; text: "Derivative" }
            }
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 5
                Text { text: "X regions"; color: root.colors.muted; font.pointSize: 8 }
                ChoiceEditor {
                    Layout.fillWidth: true
                    bridge: viewBridge
                    colors: root.colors
                    fieldKey: "regionCount"
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                visible: root.displayState.regionCount > 1
                spacing: 6
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    Text { text: "Boundary 1"; color: root.colors.text; font.pointSize: 9 }
                    NumberEditor {
                        Layout.fillWidth: true
                        bridge: viewBridge
                        colors: root.colors
                        fieldKey: "boundary.x0"
                    }
                    FlagEditor { bridge: viewBridge; colors: root.colors; fieldKey: "fix.boundary.x0"; compact: true }
                }
                RowLayout {
                    Layout.fillWidth: true
                    visible: root.displayState.regionCount === 3
                    spacing: 6
                    Text { text: "Boundary 2"; color: root.colors.text; font.pointSize: 9 }
                    NumberEditor {
                        Layout.fillWidth: true
                        bridge: viewBridge
                        colors: root.colors
                        fieldKey: "boundary.x1"
                    }
                    FlagEditor { bridge: viewBridge; colors: root.colors; fieldKey: "fix.boundary.x1"; compact: true }
                }
                FlagEditor {
                    Layout.fillWidth: true
                    bridge: viewBridge
                    colors: root.colors
                    fieldKey: "boundary.show"
                }
            }
            // Both panels stay alive when values or the selected product change.
            // A snapshot refresh must never replace an active numeric editor.
            ColorEditor {
                Layout.fillWidth: true
                visible: root.displayState.colorProduct === 0
                bridge: viewBridge
                colors: root.colors
                product: "raw"
            }
            ColorEditor {
                Layout.fillWidth: true
                visible: root.displayState.colorProduct === 1
                bridge: viewBridge
                colors: root.colors
                product: "second"
            }

            Divider { }
            Basic.Button {
                id: moreButton
                objectName: "drr_more_display_settings"
                Layout.fillWidth: true
                implicitHeight: 44
                leftPadding: 0
                rightPadding: 8
                checkable: true
                checked: false
                hoverEnabled: true
                focusPolicy: Qt.StrongFocus
                Accessible.name: "More display settings"
                Accessible.description: checked ? "Expanded" : "Collapsed"
                contentItem: RowLayout {
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 3
                        Text {
                            text: "More display settings"
                            color: root.colors.text
                            font.pointSize: 10
                            font.weight: Font.DemiBold
                        }
                        Text {
                            text: "Scale, secondary axis and saved defaults"
                            color: root.colors.muted
                            font.pointSize: 8
                        }
                    }
                    Canvas {
                        Layout.preferredWidth: 10
                        Layout.preferredHeight: 10
                        property bool expanded: moreButton.checked
                        property color ink: root.colors.muted
                        onExpandedChanged: requestPaint()
                        onInkChanged: requestPaint()
                        onPaint: {
                            let context = getContext("2d")
                            context.reset()
                            context.strokeStyle = ink
                            context.lineWidth = 1.3
                            context.lineCap = "round"
                            context.beginPath()
                            context.moveTo(2, expanded ? 6 : 3)
                            context.lineTo(5, expanded ? 3 : 6)
                            context.lineTo(8, expanded ? 6 : 3)
                            context.stroke()
                        }
                    }
                }
                background: Rectangle {
                    radius: 6
                    color: moreButton.down ? root.colors.pressed : moreButton.hovered ? root.colors.hover : "transparent"
                    border.width: moreButton.visualFocus ? 2 : 0
                    border.color: root.colors.focus
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                visible: moreButton.checked
                spacing: 8
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 5
                    Text { text: "Right Y-axis"; color: root.colors.muted; font.pointSize: 8 }
                    ChoiceEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "yaxis.right" }
                }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "log" }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "clip" }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "zero" }
                SectionTitle {
                    Layout.fillWidth: true
                    Layout.topMargin: 6
                    text: "Remember display settings"
                    caption: "Choose which settings follow files and centers."
                }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "memory.files" }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "memory.centers" }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "memory.colors" }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "memory.global" }
                FlagEditor { Layout.fillWidth: true; bridge: viewBridge; colors: root.colors; fieldKey: "memory.lock_x" }
                DisplayButton {
                    objectName: "drr_action_memory.auto"
                    Layout.fillWidth: true
                    colors: root.colors
                    text: (root.displayState.actions["memory.auto"] || {}).text || "Restore Auto for this load"
                    enabled: (root.displayState.actions["memory.auto"] || {}).enabled !== false
                    tooltip: (root.displayState.actions["memory.auto"] || {}).tooltip || ""
                    onClicked: root.bridge.invoke("memory.auto")
                }
                DisplayButton {
                    objectName: "drr_action_memory.clear"
                    Layout.fillWidth: true
                    colors: root.colors
                    text: (root.displayState.actions["memory.clear"] || {}).text || "Clear this center template"
                    enabled: (root.displayState.actions["memory.clear"] || {}).enabled !== false
                    tooltip: (root.displayState.actions["memory.clear"] || {}).tooltip || ""
                    onClicked: root.bridge.invoke("memory.clear")
                }
                Text {
                    objectName: "drr_fix_source_legend"
                    Layout.fillWidth: true
                    text: (root.displayState.texts.fixLegend || {}).text || ""
                    color: root.colors.muted
                    font.pointSize: 8
                    wrapMode: Text.Wrap
                }
            }
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: memoryStatus.implicitHeight + 18
                visible: memoryStatus.text.length > 0
                radius: 5
                color: root.colors.inset
                Text {
                    id: memoryStatus
                    objectName: "drr_memory_status"
                    anchors.fill: parent
                    anchors.margins: 9
                    text: (root.displayState.texts.memoryStatus || {}).text || ""
                    color: root.colors.muted
                    font.pointSize: 8
                    wrapMode: Text.Wrap
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
            }
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: errorMessage.implicitHeight + 18
                visible: errorMessage.text.length > 0
                radius: 5
                color: root.colors.errorSurface
                border.color: root.colors.error
                Text {
                    id: errorMessage
                    objectName: "drr_display_error"
                    anchors.fill: parent
                    anchors.margins: 9
                    text: root.bridge.error || ""
                    color: root.colors.error
                    font.pointSize: 8
                    wrapMode: Text.Wrap
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
            }
            Item { Layout.preferredHeight: 4 }
        }
    }
}
