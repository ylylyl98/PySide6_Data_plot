' Explorer open command. Open the document even if the optional plot editor is unavailable.
Option Explicit
Dim args, fs, shell, deckcraft, pythonw, editor, document
Set args = WScript.Arguments
If args.Count <> 4 Then WScript.Quit 2
deckcraft = args(0)
pythonw = args(1)
editor = args(2)
document = args(3)
Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("Shell.Application")

Function Quoted(value)
    Quoted = Chr(34) & value & Chr(34)
End Function

If LCase(fs.GetExtensionName(document)) = "pptx" Then
    ' Port zero gives each DeckCraft instance its own Windows-assigned loopback port.
    shell.ShellExecute deckcraft, "--control 0 " & Quoted(document), fs.GetParentFolderName(deckcraft), "open", 1
    If fs.FileExists(pythonw) And fs.FileExists(editor) Then
        shell.ShellExecute pythonw, Quoted(editor) & " --deckcraft-companion", fs.GetParentFolderName(editor), "open", 7
    End If
Else
    shell.ShellExecute deckcraft, Quoted(document), fs.GetParentFolderName(deckcraft), "open", 1
End If
