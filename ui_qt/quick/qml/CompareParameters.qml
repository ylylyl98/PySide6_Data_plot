pragma ComponentBehavior: Bound

import QtQuick

AppParameterPane {
    id: pane
    objectName: "quick_compare_parameters"
    readonly property var state: bridge.snapshot
    groups: state.parameterGroups || []
    message: state.error || ""
    colorRangeController: bridge.colorRangeController
    Accessible.name: "Compare parameters"
}
