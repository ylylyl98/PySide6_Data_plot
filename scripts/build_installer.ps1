param(
    [Parameter(Mandatory = $true)][ValidatePattern('^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$')][string]$Version,
    [string]$OutputFile = "DPTK-Setup-Windows-x64.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo "dist\PySide6_Data_Plot\PySide6_Data_Plot.exe"
$script = Join-Path $repo "packaging\installer.nsi"
$output = [System.IO.Path]::GetFullPath((Join-Path $repo $OutputFile))
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Build the PyInstaller application first: $exe" }

$makensis = (Get-Command makensis.exe -ErrorAction SilentlyContinue).Source
if (-not $makensis) {
    $candidate = Join-Path ${env:ProgramFiles(x86)} "NSIS\makensis.exe"
    if (Test-Path -LiteralPath $candidate) { $makensis = $candidate }
}
if (-not $makensis) { throw "NSIS makensis.exe was not found." }

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
Push-Location $repo
try {
    & $makensis "/DAPP_VERSION=$Version" "/DOUTPUT_FILE=$output" $script
    if ($LASTEXITCODE -ne 0) { throw "makensis failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
if (-not (Test-Path -LiteralPath $output -PathType Leaf)) { throw "Installer was not produced: $output" }
Write-Host "Built: $output"
