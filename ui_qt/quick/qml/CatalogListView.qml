pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic

ListView {
    id: view
    property var scrollAnchor: null
    property int restoreGeneration: 0

    function revealId(identity) {
        const index = model && identity ? model.indexOfId(identity) : -1
        if (index < 0)
            return false
        // Reopening is an explicit navigation request; an older model-reset
        // timer must not restore a different viewport after this one.
        scrollAnchor = null
        ++restoreGeneration
        currentIndex = index
        forceLayout()
        positionViewAtIndex(index, ListView.Center)
        return true
    }

    function rememberPosition() {
        ++restoreGeneration
        // Inventory preview and final history can reset the model in one turn.
        // Keep the original position until the newest reset has been laid out.
        if (scrollAnchor)
            return
        if (atYBeginning || !model || count === 0)
            return
        // The viewport can begin in the gap between two variable-height rows.
        for (let delta = 1; delta <= spacing + 16; delta += 2) {
            const index = indexAt(contentX + leftMargin + 1, contentY + delta)
            const item = index >= 0 ? itemAtIndex(index) : null
            if (item && item.entry) {
                scrollAnchor = { id: item.entry.id, offset: item.y - contentY }
                return
            }
        }
    }

    onModelChanged: { scrollAnchor = null; ++restoreGeneration }
    onMovementStarted: { scrollAnchor = null; ++restoreGeneration }

    function restorePosition(generation) {
        if (generation !== restoreGeneration || !scrollAnchor || !model)
            return
        const index = model.indexOfId(scrollAnchor.id)
        if (index < 0) {
            scrollAnchor = null
            return
        }
        positionViewAtIndex(index, ListView.Beginning)
        forceLayout()
        const item = itemAtIndex(index)
        if (item) {
            const firstY = originY - topMargin
            const lastY = Math.max(firstY, originY + contentHeight - height + bottomMargin)
            contentY = Math.max(firstY, Math.min(lastY, item.y - scrollAnchor.offset))
        }
        scrollAnchor = null
    }

    Timer {
        id: restoreTimer
        interval: 0
        property int generation: 0
        onTriggered: view.restorePosition(generation)
    }
    Connections {
        target: view.model
        function onModelAboutToBeReset() { view.rememberPosition() }
        function onModelReset() {
            restoreTimer.generation = view.restoreGeneration
            restoreTimer.restart()
        }
    }
    Basic.ScrollBar.vertical: Basic.ScrollBar {}
}
