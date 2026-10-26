pragma ComponentBehavior: Bound

import QtQuick

AppParameterPane {
    id: pane
    objectName: "quick_shg_parameters"
    readonly property var state: bridge ? bridge.snapshot : ({})
    groups: state.parameterGroups || []
    message: state.error || ""
    Accessible.name: "SHG processing, background and angular-fit parameters"
}
