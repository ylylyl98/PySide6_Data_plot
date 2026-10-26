pragma ComponentBehavior: Bound
import QtQuick

AppParameterPane {
    id: pane
    objectName: "quick_mcd_parameters"
    readonly property var state: bridge ? bridge.snapshot : ({})
    groups: state.parameterGroups || []
    message: state.error || ""
    Accessible.name: "MCD processing, backgrounds, feature analysis and retained results"
}
