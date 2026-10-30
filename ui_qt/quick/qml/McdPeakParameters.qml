pragma ComponentBehavior: Bound
import QtQuick

AppParameterPane {
    id: pane
    objectName: "quick_mcd_peak_parameters"
    readonly property var state: bridge ? bridge.viewState : ({})
    groups: state.parameterGroups || []
    message: state.error || ""
    Accessible.name: "MCD reflection peak source, detection, model and export parameters"
}
