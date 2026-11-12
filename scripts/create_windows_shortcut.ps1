[CmdletBinding()]
param(
    [string]$ShortcutPath,
    [switch]$Desktop
)

# Compatibility entry; Data_Plot_App.bat is the normal first-run path.
$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Run Data_Plot_App.bat first to prepare the source environment."
}
Push-Location -LiteralPath $projectRoot
try {
    if ($ShortcutPath) {
        & $python -c 'import sys; from pathlib import Path; from windows_source import create_shortcut; print(create_shortcut(Path(sys.argv[1])))' $ShortcutPath
    } else {
        & $python -c 'from windows_source import create_shortcut; print(create_shortcut())'
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the DPTK Source shortcut." }
} finally {
    Pop-Location
}
