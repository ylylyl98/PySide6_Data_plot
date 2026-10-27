pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Dptk 1.0

Rectangle {
    id: root
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    readonly property var referenceFiles: state.method === "matched" ? (state.selectedCandidates || []) : (state.files || [])
    property var invalidFields: ({})
    property bool showReport: false
    readonly property bool advancedPage: state.page !== "settings"
    readonly property bool inlineReferencesVisible: !advancedPage && ["bn", "p", "n", "pn", "custom"].indexOf(state.kind) >= 0
    readonly property bool hasReport: !!state.texts?.report?.length
    readonly property bool previewPaneVisible: advancedPage || (showReport && hasReport)
    readonly property var sourceTypes: [
        {key: "self", title: "Self", choice: 4},
        {key: "bn", title: "BN", choice: 0},
        {key: "p", title: "P", choice: 1},
        {key: "n", title: "N", choice: 2},
        {key: "pn", title: "P + N", choice: 3},
        {key: "fitted", title: "Fitted", choice: 5}
    ]
    readonly property int selectedSourceTab: sourceTypes.findIndex(source => source.key === state.kind)
    readonly property string sourceTitle: state.kind === "self" ? "Self reference"
        : state.kind === "custom" ? "Current custom reference"
        : (sourceTypes.find(source => source.key === state.kind)?.title || "Current") + " background"
    readonly property var toolPages: [
        {key: "pn", title: "P / N drafts and saved versions"},
        {key: "fit", title: "Fit sample background"},
        {key: "composite", title: "Build composite"},
        {key: "bn", title: "Compare BN references"},
        {key: "raw", title: "Preview original data"}
    ]
    readonly property var availableTools: toolPages.filter(tool => tool.key === "raw"
        || tool.key === "pn" && ["p", "n", "pn"].indexOf(state.kind) >= 0
        || tool.key === "fit" && state.kind === "fitted"
        || tool.key === "bn" && state.kind === "bn"
        || tool.key === "composite" && ["bn", "p", "n", "pn", "custom"].indexOf(state.kind) >= 0)
    readonly property bool invalid: Object.keys(invalidFields).length > 0
    implicitWidth: 1060
    implicitHeight: 720
    color: colors.surface
    objectName: "drr_background_workspace"
    onInlineReferencesVisibleChanged: if (bridge) bridge.setInlineSelectionVisible(inlineReferencesVisible && visible)
    onVisibleChanged: if (bridge) bridge.setInlineSelectionVisible(inlineReferencesVisible && visible)
    Component.onCompleted: bridge.setInlineSelectionVisible(inlineReferencesVisible && visible)
    Component.onDestruction: if (bridge) bridge.setInlineSelectionVisible(false)

    function activateSource(index) {
        let source = sourceTypes[index]
        if (!source || state.busy || invalid) return
        // A repeated selection must not invalidate an existing BN preview.
        if (state.kind !== source.key) bridge.setChoice("kind", source.choice)
        if (state.page !== "settings") bridge.invoke("page:settings")
    }
    function focusSource(index) {
        index = Math.max(0, Math.min(sourceTypes.length - 1, index))
        activateSource(index)
        sourceTabs.itemAt(index).forceActiveFocus(Qt.TabFocusReason)
    }

    function invalidField(key, value) {
        let next = Object.assign({}, invalidFields)
        if (value) next[key] = true
        else delete next[key]
        invalidFields = next
    }

    component Note: Text {
        Layout.fillWidth: true
        color: root.colors.muted
        wrapMode: Text.Wrap
        font.pointSize: 9
    }
    component Heading: Text {
        Layout.fillWidth: true
        color: root.colors.text
        font.weight: Font.DemiBold
        font.pointSize: 10
        wrapMode: Text.Wrap
    }
    component Action: AppButton {
        required property string command
        colors: root.colors
        compact: true
        enabled: !root.state.busy && !root.invalid
        onClicked: root.bridge.invoke(command)
        objectName: "drr_background_" + command.replace(/:/g, "_")
    }
    component Choice: GridLayout {
        id: choiceGroup
        required property string field
        required property string label
        property bool inlineLabel: false
        property var descriptor: root.state.choices ? root.state.choices[field] : ({items: [], current: -1})
        Layout.fillWidth: true
        columns: inlineLabel ? 2 : 1
        rowSpacing: 3
        columnSpacing: 6
        Note { Layout.fillWidth: !choiceGroup.inlineLabel; text: choiceGroup.label }
        AppComboBox {
            colors: root.colors
            id: combo
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            implicitHeight: 30
            model: parent.descriptor.items
            currentIndex: parent.descriptor.current
            enabled: !root.state.busy
            Accessible.name: parent.label
            objectName: "drr_background_choice_" + parent.field
            onActivated: index => root.bridge.setChoice(parent.field, index)
            contentItem: Text {
                text: combo.displayText
                color: combo.enabled ? root.colors.text : root.colors.disabled
                font.pointSize: 9
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                leftPadding: 7
                rightPadding: 24
            }
            background: Rectangle { radius: 4; color: root.colors.control; border.color: combo.activeFocus ? root.colors.focus : root.colors.border }
            popup: Basic.Popup {
                y: combo.height
                width: combo.width
                implicitHeight: Math.min(300, list.contentHeight + 2)
                padding: 1
                background: Rectangle { color: root.colors.control; border.color: root.colors.border; radius: 4 }
                contentItem: ListView {
                    id: list
                    clip: true
                    implicitHeight: contentHeight
                    model: combo.delegateModel
                    currentIndex: combo.highlightedIndex
                    Basic.ScrollIndicator.vertical: Basic.ScrollIndicator { }
                }
            }
            delegate: Basic.ItemDelegate {
                required property int index
                required property var modelData
                width: combo.width
                implicitHeight: Math.max(30, contentItem.implicitHeight + 12)
                highlighted: combo.highlightedIndex === index
                background: Rectangle { color: parent.highlighted ? root.colors.hover : root.colors.control }
                contentItem: Text { text: parent.modelData; color: root.colors.text; wrapMode: Text.Wrap; font.pointSize: 9 }
            }
        }
    }
    component Flag: Basic.CheckBox {
        required property string field
        property string hint: ""
        hoverEnabled: true
        Basic.ToolTip.visible: hovered && hint.length > 0
        Basic.ToolTip.text: hint
        Accessible.description: hint
        Layout.fillWidth: true
        checked: root.state.flags ? root.state.flags[field] === true : false
        enabled: !root.state.busy
        onToggled: root.bridge.setFlag(field, checked)
        objectName: "drr_background_flag_" + field
        contentItem: Text {
            text: parent.text
            color: parent.enabled ? root.colors.text : root.colors.disabled
            wrapMode: Text.Wrap
            leftPadding: parent.indicator.width + 8
            verticalAlignment: Text.AlignVCenter
            font.pointSize: 9
        }
    }
    component Editor: ColumnLayout {
        id: editor
        required property string field
        required property string label
        property bool numeric: false
        property string placeholder: ""
        readonly property string value: String((numeric ? root.state.numbers : root.state.texts)?.[field] ?? "")
        Layout.fillWidth: true
        spacing: 3
        Note { text: editor.label }
        Basic.TextField {
            id: input
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            implicitHeight: 30
            property bool invalid: false
            property bool dirty: false
            placeholderText: editor.placeholder
            color: root.colors.text
            placeholderTextColor: root.colors.muted
            selectionColor: root.colors.accent
            selectedTextColor: root.colors.selectedText
            selectByMouse: true
            font.pointSize: 9
            enabled: !root.state.busy
            objectName: "drr_background_editor_" + editor.field
            Accessible.name: editor.label
            text: editor.value
            onTextEdited: dirty = true
            onEditingFinished: {
                if (!dirty) return
                let value = Number(text)
                invalid = editor.numeric && (!text.trim().length || !isFinite(value))
                root.invalidField(editor.field, invalid)
                if (invalid) return
                dirty = false
                if (editor.numeric) root.bridge.setNumber(editor.field, value)
                else root.bridge.setText(editor.field, text)
            }
            background: Rectangle { radius: 4; color: root.colors.control; border.width: input.activeFocus ? 2 : 1; border.color: input.invalid || (root.state.invalidNumberFields || []).indexOf(editor.field) >= 0 ? root.colors.error : input.activeFocus ? root.colors.focus : root.colors.border }
            Connections {
                target: editor
                function onValueChanged() { if (!input.dirty && !input.activeFocus) input.text = editor.value }
            }
        }
    }

    component SettingsPage: ColumnLayout {
        objectName: "drr_background_settings_page"
        spacing: 8
        GridLayout {
            Layout.fillWidth: true
            columns: root.width >= 680 ? 2 : 1
            columnSpacing: 12
            RowLayout {
                id: selfFrames
                objectName: "drr_background_self_frames"
                Layout.fillWidth: true
                visible: root.state.kind === "self"
                spacing: 6
                function selectFrame(index, focus) {
                    if (root.state.busy || root.invalid) return
                    index = Math.max(0, Math.min(1, index))
                    if (root.state.choices?.frame?.current !== index)
                        root.bridge.setChoice("frame", index)
                    if (focus) frameButtons.itemAt(index).forceActiveFocus(Qt.TabFocusReason)
                }
                Note { Layout.fillWidth: false; text: "Reference frame" }
                Repeater {
                    id: frameButtons
                    model: ["First frame", "Last frame"]
                    delegate: AppButton {
                        required property int index
                        required property string modelData
                        objectName: "drr_background_self_frame_" + (index === 0 ? "first" : "last")
                        colors: root.colors
                        text: modelData
                        description: "Use the " + (index === 0 ? "first" : "last") + " measurement frame as the Self reference after Apply."
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        checkable: true
                        autoExclusive: true
                        checked: root.state.choices?.frame?.current === index
                        enabled: !root.state.busy && !root.invalid
                        Accessible.role: Accessible.RadioButton
                        Accessible.name: modelData
                        onClicked: selfFrames.selectFrame(index, false)
                        Keys.priority: Keys.BeforeItem
                        Keys.onLeftPressed: selfFrames.selectFrame(index - 1, true)
                        Keys.onRightPressed: selfFrames.selectFrame(index + 1, true)
                    }
                }
            }
            Note {
                id: frameSummary
                objectName: "drr_background_frame_summary"
                visible: root.inlineReferencesVisible
                text: root.state.choices?.frame?.current === 2
                    ? "Average all frames · Equal file weight"
                    : (root.state.choices?.frame?.items[root.state.choices.frame.current] || "") + " · Equal file weight"
                Accessible.name: text
                Basic.ToolTip.visible: frameHover.hovered
                Basic.ToolTip.text: (root.state.choices?.frame?.current === 2
                    ? "All frames are averaged within each file, then files receive equal weight. This does not match frames by gate condition. "
                    : "Use the selected first or last frame from each file, then weight files equally. ")
                    + "Change special frame settings in Advanced tools > Reference frames."
                HoverHandler { id: frameHover }
            }
            Choice { field: "method"; inlineLabel: true; label: "Use files"; visible: ["bn", "p", "n", "pn"].indexOf(root.state.kind) >= 0 }
        }
        Loader {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 180
            visible: root.inlineReferencesVisible
            active: root.inlineReferencesVisible
            sourceComponent: BackgroundFileBrowser {
                bridge: root.bridge.inlineFileSelectionBridge
                colors: root.colors
                chosenTitle: root.state.method === "matched" ? "Candidate pool" : "Selected references"
                enabled: !root.state.busy && !root.invalid
            }
        }
        GridLayout {
            Layout.fillWidth: true
            columns: root.width >= 680 ? 2 : 1
            columnSpacing: 12
            Flag { field: "fixed"; text: "Keep background across groups"; implicitHeight: 32; padding: 2; hint: "Reuse the chosen common references when changing measurement groups."; visible: root.inlineReferencesVisible; enabled: !root.state.busy && root.state.canFix }
            Flag { field: "auto_bn_compare"; text: "Automatic BN comparison"; implicitHeight: 32; padding: 2; hint: root.state.autoCompareSupported ? (root.state.flags.fixed ? "Compare for review while keeping the chosen background. Apply a candidate explicitly in BN comparison." : "Apply a candidate only after the current group's joint DR/R + D2E checks pass.") : "Unavailable in this session. Use Advanced tools to review BN references."; visible: root.state.kind === "bn"; enabled: !root.state.busy && root.state.autoCompareSupported }
        }
        Basic.ScrollView {
            id: settingsOptions
            objectName: "drr_background_settings_options"
            visible: !root.inlineReferencesVisible || root.state.method === "matched" || root.state.hasSavedAssignments === true || (root.state.warnings || []).length > 0
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(optionsContent.implicitHeight + 6, root.height * .24)
            contentWidth: availableWidth
            clip: true
            Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
            ColumnLayout {
                id: optionsContent
                width: settingsOptions.availableWidth - 8
                spacing: 5
                Note { text: "The current custom reference is retained until you apply a new draft."; visible: root.state.kind === "custom" }
                Note { text: "Restored each measurement's saved background. Changing the file selection replaces this mapping."; visible: root.state.hasSavedAssignments === true }
                Note { text: "Match group uses the experiment's BN candidates. Select files above to set an explicit candidate pool."; visible: root.state.method === "matched" && root.state.candidatePoolAutomatic }
                RowLayout {
                    Layout.fillWidth: true
                    visible: root.state.method === "matched"
                    Action { command: "match"; text: "Match group"; primary: true }
                    Note { text: "All measurements remain in the average. Review the mapping before applying." }
                }
                Repeater {
                    model: root.state.mappings || []
                    delegate: ColumnLayout {
                        id: mapping
                        required property var modelData
                        Layout.fillWidth: true
                        Note { text: mapping.modelData.measurement + " → " + mapping.modelData.background + "\n" + mapping.modelData.detail }
                        AppComboBox {
                            colors: root.colors
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                            visible: mapping.modelData.canOverride
                            model: mapping.modelData.candidates
                            currentIndex: mapping.modelData.candidates.indexOf(mapping.modelData.background)
                            onActivated: index => root.bridge.overrideMatch(mapping.modelData.measurement, mapping.modelData.candidates[index])
                            Accessible.name: "Manual reference for " + mapping.modelData.measurement
                        }
                    }
                }
                Note { text: (root.state.warnings || []).join("\n"); visible: text.length > 0 }
                Flag { field: "acknowledged"; text: "I reviewed the matching warnings"; visible: (root.state.warnings || []).length > 0 }
                Action { command: "page:fit"; text: "Fit / view sample background…"; visible: root.state.kind === "fitted" }
                Note { text: "Self uses the selected first or last frame in the loaded data, before ratio and derivative processing."; visible: root.state.kind === "self" }
            }
        }
        Basic.ScrollView {
            objectName: "drr_background_settings_report"
            Layout.fillWidth: true
            Layout.preferredHeight: 120
            visible: root.showReport && root.hasReport
            contentWidth: availableWidth
            clip: true
            Basic.TextArea { text: root.state.texts?.report || ""; readOnly: true; selectByMouse: true; wrapMode: Text.Wrap; color: root.colors.text; font.pointSize: 8; background: Rectangle { color: root.colors.inset } }
        }
        Item { Layout.fillHeight: true; visible: !root.inlineReferencesVisible }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8
        RowLayout {
            id: sourceTabs
            objectName: "drr_background_source_tabs"
            Layout.fillWidth: true
            spacing: 4
            readonly property int currentIndex: root.selectedSourceTab
            function itemAt(index) { return sourceButtons.itemAt(index) }
            Accessible.role: Accessible.PageTabList
            Accessible.name: "Background source"
            Repeater {
                id: sourceButtons
                model: root.sourceTypes
                delegate: AppTabButton {
                    required property var modelData
                    required property int index
                    objectName: "drr_background_source_" + modelData.key
                    colors: root.colors
                    text: modelData.title
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    Layout.preferredHeight: 32
                    Layout.minimumWidth: 0
                    autoExclusive: true
                    checked: root.state.kind === modelData.key
                    enabled: !root.state.busy && !root.invalid
                    Accessible.name: modelData.title + " background source"
                    onClicked: root.activateSource(index)
                    Keys.priority: Keys.BeforeItem
                    Keys.onLeftPressed: root.focusSource(index - 1)
                    Keys.onRightPressed: root.focusSource(index + 1)
                    Keys.onPressed: function(event) {
                        if (event.key === Qt.Key_Home || event.key === Qt.Key_End) {
                            root.focusSource(event.key === Qt.Key_Home ? 0 : root.sourceTypes.length - 1)
                            event.accepted = true
                        }
                    }
                }
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Heading { Layout.fillWidth: false; text: root.advancedPage ? (root.toolPages.find(page => page.key === root.state.page)?.title || "Background tools") : root.sourceTitle }
            Note { Layout.fillWidth: false; text: root.state.measurementSummary || "" }
            Item { Layout.fillWidth: true }
            Action { command: "page:settings"; text: "Back to background"; visible: root.advancedPage }
            AppButton {
                objectName: "drr_background_diagnostics"
                colors: root.colors
                compact: true
                text: root.showReport ? "Hide diagnostics" : "Diagnostics"
                visible: root.hasReport
                description: "Show the last background preview diagnostics"
                onClicked: root.showReport = !root.showReport
            }
            AppButton {
                id: toolsButton
                objectName: "drr_background_tools"
                colors: root.colors
                text: "Advanced tools…"
                description: "Generate, fit or inspect a background"
                enabled: !root.state.busy && !root.invalid
                onClicked: toolsMenu.open()
                Basic.Menu {
                    id: toolsMenu
                    objectName: "drr_background_tools_menu"
                    y: toolsButton.height
                    Basic.Menu {
                        title: "Reference frames"
                        objectName: "drr_background_reference_frames_menu"
                        enabled: ["bn", "p", "n", "pn", "custom"].indexOf(root.state.kind) >= 0
                        Repeater {
                            model: root.state.choices?.frame?.items || []
                            delegate: Basic.MenuItem {
                                required property int index
                                required property var modelData
                                objectName: "drr_background_reference_frame_" + index
                                text: index === 2 ? "Average all frames (default)" : modelData
                                checkable: true
                                checked: root.state.choices?.frame?.current === index
                                enabled: !root.state.busy && !root.invalid
                                onTriggered: root.bridge.setChoice("frame", index)
                            }
                        }
                    }
                    Repeater {
                        model: root.availableTools
                        delegate: Basic.MenuItem {
                            required property var modelData
                            objectName: "drr_background_tool_" + modelData.key
                            text: modelData.title + "…"
                            enabled: !root.state.busy && !root.invalid
                            onTriggered: root.bridge.invoke("page:" + modelData.key)
                        }
                    }
                }
            }
        }
        Loader {
            Layout.fillWidth: true
            Layout.fillHeight: true
            active: !root.advancedPage
            visible: active
            sourceComponent: SettingsPage {}
        }
        Basic.SplitView {
            visible: root.advancedPage
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: root.width < 760 ? Qt.Vertical : Qt.Horizontal
            Basic.ScrollView {
                id: controls
                objectName: "drr_background_controls"
                Basic.SplitView.fillWidth: !root.previewPaneVisible
                Basic.SplitView.preferredWidth: 320
                Basic.SplitView.minimumWidth: 250
                Basic.SplitView.preferredHeight: root.height * .53
                Basic.SplitView.minimumHeight: 120
                clip: true
                contentWidth: availableWidth
                Basic.ScrollBar.horizontal.policy: Basic.ScrollBar.AlwaysOff
                ColumnLayout {
                    width: controls.availableWidth - 12
                    spacing: 9
                    enabled: !root.state.busy
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: root.state.page === "pn"
                        Heading { text: "Independent P and N sources" }
                        Note { text: "Select one original CSV per role. Choosing a source retains the opposite role. Newest file modification time appears first; it is not acquisition chronology." }
                        Note { text: "P: " + (root.state.pFiles || []).join(", ") + "\nN: " + (root.state.nFiles || []).join(", ") }
                        Note { text: root.state.pairSummary || "" }
                        Note { text: "Choose P and N files independently. Load saved versions when you need a previously generated draft."; visible: !root.state.catalogLoaded }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "choose-files:P"; text: "Choose P files…" } Action { command: "choose-files:N"; text: "Choose N files…" } Action { command: "refresh"; text: root.state.catalogLoaded ? "Refresh catalog" : "Load catalog" } }
                        Repeater {
                            model: root.state.pnCatalog || []
                            delegate: ColumnLayout {
                                required property var modelData
                                Layout.fillWidth: true
                                AppButton { colors: root.colors; compact: true; Layout.fillWidth: true; text: modelData.role + ": " + modelData.file; primary: modelData.selected; onClicked: root.bridge.selectCatalog(modelData.role, modelData.file); description: modelData.detail }
                                Note { text: modelData.time + " · " + modelData.detail }
                            }
                        }
                        Choice { field: "generation"; label: "P/N method" }
                        Choice { field: "pFrame"; label: "P frames" }
                        Choice { field: "nFrame"; label: "N frames" }
                        Note { text: "Reconstruction uses all frames and exactly two explicitly selected training repeats. Other measurements remain checks. Generated drafts are not scientifically accepted automatically." }
                        Repeater {
                            model: root.state.training || []
                            delegate: Basic.CheckBox {
                                required property var modelData
                                Layout.fillWidth: true
                                text: "Train: " + modelData.file
                                checked: modelData.selected
                                onToggled: root.bridge.setFlag("training:" + modelData.file, checked)
                                contentItem: Text { text: parent.text; leftPadding: parent.indicator.width + 8; color: root.colors.text; wrapMode: Text.Wrap; font.pointSize: 9 }
                            }
                        }
                        Editor { field: "featureProtection"; label: "Explicit raw-source feature protection (eV)"; placeholder: "P 1.70 1.72; N 1.85 1.87" }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "generate"; text: "Generate draft" } Action { command: "regenerate"; text: "Force regenerate" } }
                        Choice { field: "version"; label: "All saved P/N versions" }
                        Note { text: root.state.versionStatus || "" }
                        Flag { field: "allow_stale"; text: "Use this stale version's saved snapshot explicitly" }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "browse:pn-import"; text: "Import saved CSV…" } Action { command: "use-version"; text: "Use in settings"; enabled: !root.state.busy && root.state.canUseVersion } }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: root.state.page === "fit"
                        Heading { text: "Fit one raw measurement average" }
                        Flag { field: "automatic_protection"; text: "Automatic gate-local peak protection" }
                        Editor { field: "regions"; label: "Manual protected energy intervals (eV)"; placeholder: "1.56 1.64; 1.69 1.77" }
                        Choice { field: "fitMethod"; label: "Smooth background model" }
                        Choice { field: "fitView"; label: "Preview product" }
                        Editor { field: "spline_intervals"; label: "Spline intervals (4–40)"; numeric: true }
                        Editor { field: "smoothing"; label: "Smoothing (0.000001–10)"; numeric: true }
                        Editor { field: "polynomial_degree"; label: "Polynomial degree (2–4)"; numeric: true }
                        Flag { field: "shared_texture"; text: "Shared fine texture correction" }
                        Flag { field: "periodic_texture"; text: "Validated periodic gap prediction" }
                        Editor { field: "periodic_components"; label: "Maximum periodic components (1–5)"; numeric: true }
                        Editor { field: "texture_reference_source"; label: "Explicit fringe reference source" }
                        Action { command: "browse:texture"; text: "Select fringe reference…" }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "suggest-fit"; text: "Suggest protection" } Action { command: "add-suggestions"; text: "Add suggestions" } Action { command: "fit"; text: "Fit preview"; primary: true } }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "check-fit"; text: "Check average" } Action { command: "compare-fit"; text: "Compare models" } Action { command: "use-fit"; text: "Use in settings"; enabled: !root.state.busy && root.state.hasFit } }
                        Choice { field: "template"; label: "Reusable recipe template" }
                        Editor { field: "templateName"; label: "Save template name" }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "load-template"; text: "Load template" } Action { command: "save-template"; text: "Save recipe template" } }
                        Note { text: "Templates contain settings only. Shared and periodic texture coverage depends on the original data and its validation; inspect the report." }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: root.state.page === "composite"
                        Heading { text: "Native-grid composite editor" }
                        Note { text: "Regions switch source intensity directly on the original ordered spectral grid. Saving creates a new CSV and recipe; managed P/N versions remain immutable." }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "browse:composite-add"; text: "Add source…" } Action { command: "browse:composite-open"; text: "Open composite…" } }
                        Flag { field: "group_composite"; text: "Average selected files as one source" }
                        Choice { field: "compositeSource"; label: "Region source" }
                        Choice { field: "frame"; label: "Frames for a new source" }
                        Action { command: "composite-reload"; text: "Reload selected original source" }
                        Editor { field: "regionLow"; label: "Region low (eV)"; numeric: true }
                        Editor { field: "regionHigh"; label: "Region high (eV)"; numeric: true }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "composite-region"; text: "Assign interval" } Action { command: "composite-default"; text: "Set default source" } Action { command: "composite-remove"; text: "Remove source" } }
                        Repeater {
                            model: root.state.regions || []
                            delegate: ColumnLayout {
                                required property var modelData
                                required property int index
                                Layout.fillWidth: true
                                Note { text: parent.modelData.low + "–" + parent.modelData.high + " eV → " + parent.modelData.source }
                                RowLayout {
                                    AppButton { colors: root.colors; compact: true; text: "Edit interval"; onClicked: root.bridge.selectRegion(parent.parent.index) }
                                    AppButton { colors: root.colors; compact: true; text: "Remove interval"; onClicked: root.bridge.removeRegion(parent.parent.index) }
                                }
                            }
                        }
                        Action { command: "composite-new-region"; text: "New interval" }
                        Flag { field: "reconstruct_gaps"; text: "Reconstruct supported short gaps" }
                        Editor { field: "maxGapMeV"; label: "Maximum reconstructed gap (meV)"; numeric: true }
                        Flag { field: "preserve_regions"; text: "Keep manual intervals in suggestions" }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "composite-undo"; text: "Undo"; enabled: !root.state.busy && root.state.canUndo } Action { command: "composite-suggest"; text: "Suggest composite" } Action { command: "composite-save"; text: "Save as / use draft…" } }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: root.state.page === "bn"
                        Heading { text: "Measured BN group comparison" }
                        Note { text: "Uses the selected common BN reference and current SG settings. Leave trusted calibration ranges empty for preview only; intervals require independent physical evidence." }
                        Repeater {
                            model: [{key: "train_region", title: "Fit energy"}, {key: "validation_region", title: "Check energy"}, {key: "core_region", title: "Correction energy"}]
                            delegate: ColumnLayout {
                                required property var modelData
                                Layout.fillWidth: true
                                Editor { field: "bn." + modelData.key + ".low"; label: modelData.title + " low (eV)"; numeric: true }
                                Editor { field: "bn." + modelData.key + ".high"; label: modelData.title + " high (eV)"; numeric: true }
                            }
                        }
                        Editor { field: "bn.taper_width"; label: "Transition width (eV)"; numeric: true }
                        Editor { field: "bn.train_regions"; label: "Optional fit intervals (eV)" }
                        Editor { field: "bn.protected_regions"; label: "Protected intervals (eV)" }
                        Editor { field: "bn.uncertain_regions"; label: "Uncertain intervals (eV)" }
                        Editor { field: "bn.trusted_regions"; label: "Trusted calibration intervals (eV)" }
                        Action { command: "compare-bn"; text: "Compare BN / save report"; primary: true }
                        Flow { Layout.fillWidth: true; spacing: 5; Action { command: "use-bn-original"; text: "Keep original in settings" } Action { command: "use-bn-candidate"; text: "Use validated candidate"; enabled: !root.state.busy && root.state.canUseCalibrated } }
                        Repeater {
                            model: root.state.bnCandidates || []
                            delegate: Note { required property var modelData; text: modelData.name + " · " + modelData.status + "\n" + modelData.detail }
                        }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: root.state.page === "raw"
                        Heading { text: "Read-only original intensity" }
                        Choice { field: "rawRole"; label: "Source role" }
                        Choice { field: "rawScope"; label: "View" }
                        Choice { field: "rawSource"; label: "Original source"; enabled: root.state.choices?.rawScope?.current === 0 }
                        Action { command: "preview-raw"; text: "Load raw preview"; primary: true }
                        Note { text: "The reference view respects the loaded per-measurement recipe and frame policy. A mean reference does not imply a common denominator for heterogeneous DRR averages." }
                    }
                    Editor { field: "gate"; label: "Preview gate / frame index"; numeric: true; visible: root.state.page === "fit" || root.state.page === "raw" }
                    Note { text: root.state.gateDescription || ""; visible: root.state.page === "fit" || root.state.page === "raw" }
                    Item { Layout.preferredHeight: 8 }
                }
            }
            ColumnLayout {
                objectName: "drr_background_preview_pane"
                visible: root.previewPaneVisible
                Basic.SplitView.fillWidth: true
                Basic.SplitView.fillHeight: true
                Basic.SplitView.minimumWidth: 180
                Basic.SplitView.minimumHeight: 180
                spacing: 6
                Flow {
                    visible: root.advancedPage
                    Layout.fillWidth: true
                    spacing: 4
                    Action { command: "plot:home"; text: "Reset view"; enabled: !root.state.busy && root.state.hasPreview }
                    Action { command: "plot:back"; text: "Back"; enabled: !root.state.busy && root.state.hasPreview }
                    Action { command: "plot:forward"; text: "Forward"; enabled: !root.state.busy && root.state.hasPreview }
                    Action { command: "plot:pan"; text: "Pan"; enabled: !root.state.busy && root.state.hasPreview; checked: (root.state.plotTool || "").indexOf("pan") === 0 }
                    Action { command: "plot:zoom"; text: "Zoom"; enabled: !root.state.busy && root.state.hasPreview; checked: (root.state.plotTool || "").indexOf("zoom") === 0 }
                }
                Item {
                    visible: root.advancedPage
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 160
                    PlotSurface { objectName: "drr_background_plot"; anchors.fill: parent; opacity: root.state.hasPreview ? 1 : 0; backend: root.bridge ? root.bridge.plotBackend : null }
                    Rectangle {
                        anchors.fill: parent
                        visible: !root.state.hasPreview
                        color: root.colors.inset
                        radius: 5
                        border.color: root.colors.divider
                        Text {
                            objectName: "drr_background_empty_preview"
                            anchors.centerIn: parent
                            width: Math.max(100, parent.width - 40)
                            text: "Run this tool's preview to inspect the background. Use Back to background to choose and apply a reference."
                            color: root.colors.muted
                            wrapMode: Text.Wrap
                            horizontalAlignment: Text.AlignHCenter
                            font.pointSize: 10
                        }
                    }
                }
                Basic.ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: !root.advancedPage
                    Layout.preferredHeight: root.state.texts?.report ? 130 : 0
                    visible: root.showReport && root.state.texts?.report?.length > 0
                    clip: true
                    contentWidth: availableWidth
                    Basic.TextArea { text: root.state.texts?.report || ""; readOnly: true; selectByMouse: true; wrapMode: Text.Wrap; color: root.colors.text; font.pointSize: 8; background: Rectangle { color: root.colors.inset } }
                }
            }
        }
        Basic.ProgressBar { Layout.fillWidth: true; indeterminate: true; visible: root.state.busy }
        Note { visible: !root.inlineReferencesVisible || root.state.busy || !!root.state.error; text: root.state.error || root.state.status || ""; color: root.state.error ? root.colors.error : root.colors.muted; maximumLineCount: 4; elide: Text.ElideRight }
        RowLayout {
            Layout.fillWidth: true
            Note { text: root.invalid ? "Correct the highlighted numeric draft before continuing." : "Previews and saved drafts leave the applied background unchanged." }
            Action { command: "cancel-task"; text: "Stop task"; visible: root.state.busy; enabled: true }
            Action { command: "cancel"; text: "Close"; description: "Close without applying this background draft"; enabled: true }
            Action { command: "apply"; text: "Apply background"; primary: true; enabled: root.state.applyEnabled && !root.invalid; objectName: "drr_background_apply" }
        }
    }
    Loader { id: fileSelectionLoader; anchors.fill: parent }
    Binding { target: fileSelectionLoader.item; property: "colors"; value: root.colors; when: fileSelectionLoader.item !== null }
    Connections {
        target: root.bridge
        function onFileSelectionRequested() {
            if (!fileSelectionLoader.item)
                fileSelectionLoader.setSource(Qt.resolvedUrl("SourceCatalog.qml"), {bridge: root.bridge.fileSelectionBridge, colors: root.colors})
            if (fileSelectionLoader.item) fileSelectionLoader.item.open()
        }
        function onCloseRequested() {
            if (fileSelectionLoader.item) fileSelectionLoader.item.close()
        }
    }
}
