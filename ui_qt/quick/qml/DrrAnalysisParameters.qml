pragma ComponentBehavior: Bound
import QtQuick

AppParameterPane {
    id: pane
    objectName: "drr_analysis_parameters"
    readonly property var state: bridge.snapshot
    groups: state.parameterGroups || []
    message: state.error || ""
    Accessible.name: "DRR analysis parameters"
}
