.pragma library

// The Qt picker prefers its retained selection over currentFolder on open().
// A new request must discard that selection before choosing its directory.
function prepareFile(dialog, folder, suggestedUrl) {
    // selectedFiles is read-only; currentFiles is Qt's public writable list.
    dialog.currentFiles = []
    if (folder && String(folder).length > 0)
        dialog.currentFolder = folder
    if (suggestedUrl && String(suggestedUrl).length > 0)
        dialog.selectedFile = suggestedUrl
}

// AppFolderDialog creates a fresh public Qt FolderDialog for each request.
// In Qt 6.11, a reused fallback FolderDialog ignores later currentFolder changes.
function prepareFolder(dialog, folder) {
    dialog.selectedFolder = ""
    if (folder && String(folder).length > 0)
        dialog.currentFolder = folder
}
