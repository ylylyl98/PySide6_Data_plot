pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Basic.Dialog {
    id: dialog
    objectName: "megasweep_movie_dialog"
    required property var bridge
    required property var colors
    readonly property var state: bridge ? bridge.snapshot : ({})
    readonly property var options: state.movieOptions || ({})
    property bool playing: false
    property double playbackStarted: 0
    property int playbackStartFrame: 0
    title: "Spectral slice video"
    modal: true
    anchors.centerIn: parent
    width: Math.min(960, parent ? parent.width - 24 : 960)
    height: Math.min(720, parent ? parent.height - 24 : 720)
    padding: 12
    background: Rectangle { color: dialog.colors.surface; border.color: dialog.colors.border; radius: 6 }
    onClosed: { dialog.playing = false; if (dialog.state.moviePreviewBusy) dialog.bridge.invoke("movie-cancel-preview") }
    Timer {
        interval: 30
        running: dialog.visible && dialog.playing && (dialog.state.moviePreviewFrameCount || 0) > 0
        repeat: true
        onTriggered: {
            let count = Number(dialog.state.moviePreviewFrameCount || 0)
            let target = dialog.playbackStartFrame + Math.floor((Date.now() - dialog.playbackStarted) * Math.max(1, Number(dialog.options.fps || 30)) / 1000)
            if (target >= count) { target = count - 1; dialog.playing = false }
            if (target !== Number(dialog.state.moviePreviewFrameIndex || 0)) dialog.bridge.setParameter("movie.previewFrame", target)
        }
    }
    contentItem: ColumnLayout {
        spacing: 8
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Basic.ScrollView {
                id: scroll
                Layout.preferredWidth: 330
                Layout.minimumWidth: 200
                Layout.fillHeight: true
                contentWidth: availableWidth
                clip: true
                enabled: !dialog.state.busy
                ColumnLayout {
                    width: scroll.availableWidth
                    spacing: 8
                    Basic.Label { Layout.fillWidth: true; text: "Measured coordinates and playback"; color: dialog.colors.text; font.weight: Font.DemiBold; wrapMode: Text.Wrap }
                    Basic.Label { Layout.fillWidth: true; text: "FPS 1–240; repetitions 1–1000; width 320–7680 and height 240–4320 (even pixels); CRF 0–51. Leave range limits empty for automatic."; color: dialog.colors.muted; wrapMode: Text.Wrap }
                    Repeater {
                        model: [
                            {key:"first", label:"First coordinate", fallback:""}, {key:"last", label:"Last coordinate", fallback:""},
                            {key:"stride", label:"Coordinate stride", fallback:1}, {key:"hold", label:"Slice hold (s)", fallback:0.25},
                            {key:"speed", label:"Playback speed", fallback:1}, {key:"fps", label:"Frames per second", fallback:30},
                            {key:"head", label:"Extra start hold (s)", fallback:0}, {key:"tail", label:"Extra end hold (s)", fallback:0},
                            {key:"repeats", label:"Repetitions", fallback:1}, {key:"width", label:"Width (px)", fallback:1400},
                            {key:"height", label:"Height (px)", fallback:960}, {key:"crf", label:"Quality (CRF)", fallback:18},
                            {key:"vmin", label:"Color minimum", fallback:""}, {key:"vmax", label:"Color maximum", fallback:""},
                            {key:"x_min", label:"Energy minimum", fallback:""}, {key:"x_max", label:"Energy maximum", fallback:""},
                            {key:"y_min", label:"Scan minimum", fallback:""}, {key:"y_max", label:"Scan maximum", fallback:""}
                        ]
                        delegate: RowLayout {
                            required property var modelData
                            Layout.fillWidth: true
                            Basic.Label { Layout.fillWidth: true; text: parent.modelData.label; color: dialog.colors.text; wrapMode: Text.Wrap }
                            Basic.TextField {
                                id: editor
                                property bool integerField: ["stride", "fps", "repeats", "width", "height", "crf"].indexOf(parent.modelData.key) >= 0
                                objectName: "megasweep_movie_" + parent.modelData.key
                                Layout.preferredWidth: 110
                                color: dialog.colors.text
                                selectByMouse: true
                                text: String(dialog.options[parent.modelData.key] === undefined || dialog.options[parent.modelData.key] === null ? parent.modelData.fallback : dialog.options[parent.modelData.key])
                                validator: RegularExpressionValidator { regularExpression: editor.integerField ? /^\d*$/ : /^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?$/ }
                                onEditingFinished: { let old = dialog.options[parent.modelData.key]; let optional = ["first", "last", "vmin", "vmax", "x_min", "x_max", "y_min", "y_max"].indexOf(parent.modelData.key) >= 0; if (text === "" && !optional) { text = String(old === undefined || old === null ? parent.modelData.fallback : old); return } let v = text === "" ? null : Number(text); if (!dialog.state.busy && (old === undefined ? null : old) !== v) dialog.bridge.setParameter("movie." + parent.modelData.key, v) }
                                background: Rectangle { color: dialog.colors.control; border.color: parent.activeFocus ? dialog.colors.focus : dialog.colors.border; radius: 4 }
                            }
                        }
                    }
                    Basic.Label { text: "Direction"; color: dialog.colors.text }
                    Basic.ComboBox {
                        objectName: "megasweep_movie_direction"
                        Layout.fillWidth: true
                        palette.button: dialog.colors.control; palette.buttonText: dialog.colors.text; palette.text: dialog.colors.text; palette.base: dialog.colors.control; palette.highlight: dialog.colors.accent; palette.highlightedText: dialog.colors.selectedText
                        model: ["forward", "reverse", "pingpong"]
                        currentIndex: Math.max(0, model.indexOf(dialog.options.direction || "forward"))
                        onActivated: dialog.bridge.setParameter("movie.direction", currentText)
                    }
                    Basic.Label { text: "Color scale"; color: dialog.colors.text }
                    Basic.ComboBox {
                        objectName: "megasweep_movie_color_mode"
                        Layout.fillWidth: true
                        palette.button: dialog.colors.control; palette.buttonText: dialog.colors.text; palette.text: dialog.colors.text; palette.base: dialog.colors.control; palette.highlight: dialog.colors.accent; palette.highlightedText: dialog.colors.selectedText
                        model: ["global", "manual", "frame"]
                        currentIndex: Math.max(0, model.indexOf(dialog.options.color_mode || "global"))
                        onActivated: dialog.bridge.setParameter("movie.color_mode", currentText)
                    }
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    color: dialog.colors.inset; border.color: dialog.colors.border
                    Image { anchors.fill: parent; anchors.margins: 4; source: dialog.state.moviePreviewUrl || ""; fillMode: Image.PreserveAspectFit; cache: false; asynchronous: true }
                    Text { anchors.centerIn: parent; visible: !dialog.state.moviePreviewUrl; text: dialog.state.moviePreviewBusy ? "Preparing preview…" : "Preview appears here"; color: dialog.colors.muted }
                }
                Basic.Label { objectName: "megasweep_movie_preview_status"; Layout.fillWidth: true; text: dialog.state.moviePreviewStatus || dialog.state.movieDuration || ""; color: dialog.colors.muted; wrapMode: Text.Wrap }
                Basic.ProgressBar { Layout.fillWidth: true; visible: dialog.state.moviePreviewBusy === true; indeterminate: true }
                RowLayout {
                    Layout.fillWidth: true
                    AppButton { objectName: "megasweep_movie_play"; colors: dialog.colors; text: dialog.playing ? "Pause" : "Play"; enabled: (dialog.state.moviePreviewFrameCount || 0) > 0; onClicked: { if (dialog.playing) { dialog.playing = false; return } let frame = Number(dialog.state.moviePreviewFrameIndex || 0); if (frame >= Number(dialog.state.moviePreviewFrameCount || 0) - 1) { frame = 0; dialog.bridge.setParameter("movie.previewFrame", 0) } dialog.playbackStartFrame = frame; dialog.playbackStarted = Date.now(); dialog.playing = true } }
                    Basic.Slider { objectName: "megasweep_movie_seek"; Layout.fillWidth: true; from: 0; to: Math.max(0, Number(dialog.state.moviePreviewFrameCount || 0) - 1); stepSize: 1; enabled: (dialog.state.moviePreviewFrameCount || 0) > 0; value: Number(dialog.state.moviePreviewFrameIndex || 0); onMoved: { dialog.playing = false; dialog.bridge.setParameter("movie.previewFrame", Math.round(value)) } }
                    Basic.Label { objectName: "megasweep_movie_time"; text: (Number(dialog.state.moviePreviewFrameIndex || 0) / Math.max(1, Number(dialog.options.fps || 30))).toFixed(2) + " / " + Number(dialog.state.movieDuration || 0).toFixed(2) + " s"; color: dialog.colors.muted }
                }
                RowLayout {
                    AppButton { objectName: "megasweep_movie_preview"; colors: dialog.colors; text: "Preview"; enabled: !dialog.state.busy && !dialog.state.moviePreviewBusy; onClicked: dialog.bridge.invoke("movie-preview") }
                    AppButton { objectName: "megasweep_movie_cancel_preview"; colors: dialog.colors; text: "Cancel preview"; visible: dialog.state.moviePreviewBusy === true; onClicked: dialog.bridge.invoke("movie-cancel-preview") }
                }
            }
        }
        Basic.Label { Layout.fillWidth: true; visible: !!dialog.state.movieError; text: dialog.state.movieError || ""; color: dialog.colors.error; wrapMode: Text.Wrap }
        RowLayout {
            Layout.alignment: Qt.AlignRight
            AppButton { colors: dialog.colors; text: "Close"; onClicked: dialog.close() }
            AppButton { objectName: "megasweep_movie_export"; colors: dialog.colors; text: "Export MP4…"; primary: true; enabled: !dialog.state.busy && !dialog.state.moviePreviewBusy; onClicked: dialog.bridge.invoke("export-video") }
        }
    }
}
