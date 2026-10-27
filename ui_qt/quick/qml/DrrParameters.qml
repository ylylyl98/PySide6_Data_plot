pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: root
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    implicitWidth: 320
    color: colors.surface
    objectName: "drr_native_parameters"

    component ParameterTab: Basic.TabButton {
        id: tab
        implicitHeight: 30
        font.pointSize: 9
        hoverEnabled: true
        focusPolicy: Qt.StrongFocus
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
            radius: 5
            color: tab.down ? root.colors.pressed : tab.checked ? root.colors.control
                            : tab.hovered ? root.colors.hover : "transparent"
            border.width: tab.visualFocus ? 2 : tab.checked ? 1 : 0
            border.color: tab.visualFocus ? root.colors.focus : root.colors.border
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: 12
            spacing: 6
            Text { text: "Data"; color: root.colors.text; font.weight: Font.DemiBold }
            Text {
                objectName: "drr_measurement_summary"
                Layout.fillWidth: true
                text: root.state.sourceSummary || "No measurements selected"
                color: root.colors.muted
                wrapMode: Text.Wrap
                font.pointSize: 8
            }
            GridLayout {
                Layout.fillWidth: true
                columns: root.width < 280 ? 2 : 3
                columnSpacing: 6
                rowSpacing: 4
                AppButton {
                    objectName: "drr_select_measurements"
                    Layout.fillWidth: true
                    Layout.minimumWidth: 80
                    colors: root.colors
                    compact: true
                    text: "Select…"
                    description: "Select measurement files"
                    enabled: !root.state.busy
                    onClicked: root.bridge.invoke("select-files")
                }
                AppButton {
                    objectName: "drr_clear_measurements"
                    colors: root.colors
                    compact: true
                    text: "Clear"
                    enabled: !root.state.busy
                    onClicked: root.bridge.invoke("clear-files")
                }
                AppButton {
                    objectName: "drr_saved_results"
                    Layout.columnSpan: root.width < 280 ? 2 : 1
                    Layout.fillWidth: true
                    colors: root.colors
                    compact: true
                    text: "Saved results…"
                    enabled: !root.state.busy
                    onClicked: root.bridge.invoke("saved-results")
                }
            }
            Text {
                objectName: "drr_background_summary"
                Layout.fillWidth: true
                text: root.state.backgroundSummary || "Background: awaiting selection"
                color: root.colors.muted
                wrapMode: Text.Wrap
                font.pointSize: 8
            }
            AppButton {
                objectName: "drr_background_settings"
                Layout.fillWidth: true
                colors: root.colors
                compact: true
                text: "Background…"
                description: "Choose background files, self reference, P/N, BN, or fitting; Apply when ready"
                enabled: !!root.state.canConfigureBackground
                onClicked: root.bridge.invoke("background-settings")
            }
            AppButton {
                objectName: "drr_cancel_bn"
                Layout.fillWidth: true
                colors: root.colors
                compact: true
                visible: !!root.state.bnBusy
                text: "Cancel BN check"
                description: "Stop automatic BN checking and retain the previous result"
                onClicked: root.bridge.invoke("cancel-bn")
            }
        }
        Basic.TabBar {
            id: tabs
            objectName: "drr_native_parameter_tabs"
            Layout.fillWidth: true
            implicitHeight: 32
            padding: 1
            spacing: 2
            background: Rectangle { radius: 6; color: root.colors.inset; border.color: root.colors.divider }
            ParameterTab { objectName: "drr_native_display_tab"; text: "Display" }
            ParameterTab { objectName: "drr_native_processing_tab"; text: "Processing" }
            onCurrentIndexChanged: if (root.bridge && root.bridge.drrDisplayBridge)
                                       root.bridge.drrDisplayBridge.setActive(currentIndex === 0 && root.visible)
        }
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabs.currentIndex
            Display {
                bridge: root.bridge.drrDisplayBridge
                colors: root.colors
            }
            AppParameterPane {
                objectName: "drr_native_processing"
                bridge: root.bridge
                colors: root.colors
                groups: root.state.processingGroups || []
                message: root.state.processingMessage || ""
            }
        }
    }
    onVisibleChanged: if (bridge && bridge.drrDisplayBridge)
                         bridge.drrDisplayBridge.setActive(visible && tabs.currentIndex === 0)
    Component.onDestruction: if (bridge && bridge.drrDisplayBridge)
                                 bridge.drrDisplayBridge.setActive(false)
}
