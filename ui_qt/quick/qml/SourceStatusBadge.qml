pragma ComponentBehavior: Bound
import QtQuick

Rectangle {
    id: badge
    required property var colors
    property string status: "unknown"
    property string text: status === "processed" ? "Processed"
                        : status === "partial" ? "Partly processed"
                        : status === "new" ? "Unprocessed"
                        : status === "saved" ? "Saved result" : "History unknown"
    readonly property string tone: status === "processed" ? "Processed"
                                   : status === "saved" ? "Saved"
                                   : status === "new" || status === "partial" ? "New" : ""
    readonly property color foreground: colors["source" + tone + "Text"] || colors.text
    color: colors["source" + tone + "Surface"] || colors.control
    border.color: colors["source" + tone + "Border"] || colors.border
    border.width: 1
    radius: 4
    implicitWidth: label.implicitWidth + 14
    implicitHeight: label.implicitHeight + 8
    Accessible.role: Accessible.StaticText
    Accessible.name: text
    Text {
        id: label
        anchors.fill: parent
        anchors.leftMargin: 7
        anchors.rightMargin: 7
        text: badge.text
        color: badge.foreground
        font.pixelSize: 11
        font.weight: Font.DemiBold
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
