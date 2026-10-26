"""Assertions for the actual text selection colors inherited by QML windows."""


def assert_text_selection_palette(case, window, colors):
    def visual_items(parent):
        yield parent
        for child in parent.childItems():
            yield from visual_items(child)

    inputs = [item for item in visual_items(window.contentItem())
              if item.metaObject().indexOfProperty("selectionColor") >= 0
              and item.metaObject().indexOfProperty("selectedTextColor") >= 0
              and item.property("visible")]
    case.assertTrue(inputs, "The populated window must expose native text inputs")
    for item in inputs:
        label = item.objectName() or item.metaObject().className()
        case.assertEqual(item.property("selectionColor").name().lower(), colors["accent"].lower(), label)
        case.assertEqual(item.property("selectedTextColor").name().lower(), colors["selectedText"].lower(), label)
