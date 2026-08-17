[CmdletBinding()]
param(
    [switch]$SkipSync
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$codexRoot = Join-Path $env:USERPROFILE ".codex"
$openCodexRoot = Join-Path $env:USERPROFILE ".opencodex"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$agentFiles = @(
    (Join-Path $codexRoot "agents\deepseek-explorer.toml"),
    (Join-Path $codexRoot "agents\deepseek-worker.toml")
)
$openCodexConfig = Join-Path $openCodexRoot "config.json"
$catalogFiles = @(
    (Join-Path $codexRoot "opencodex-catalog.json"),
    (Join-Path $codexRoot "models_cache.json")
)

foreach ($path in ($agentFiles + $openCodexConfig)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required OpenCodex file was not found: $path"
    }
}

function Write-Utf8NoBom([string]$Path, [string]$Value) {
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Value, $encoding)
}

foreach ($path in ($agentFiles + $openCodexConfig)) {
    $backup = "$path.bak-$timestamp"
    Copy-Item -LiteralPath $path -Destination $backup -Force
    Write-Output "Backup: $backup"
}

foreach ($path in $agentFiles) {
    $content = Get-Content -Raw -LiteralPath $path
    $updated = $content.Replace("Deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash").Replace("Deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro")
    Write-Utf8NoBom -Path $path -Value $updated
}

$config = Get-Content -Raw -LiteralPath $openCodexConfig | ConvertFrom-Json
$deepseek = $config.providers.deepseek
if ($null -eq $deepseek) {
    throw "The deepseek provider was not found in $openCodexConfig"
}
if ($null -ne $deepseek.PSObject.Properties["modelSupportsReasoningSummaries"]) {
    $deepseek.PSObject.Properties.Remove("modelSupportsReasoningSummaries")
}
Write-Utf8NoBom -Path $openCodexConfig -Value ($config | ConvertTo-Json -Depth 100)

foreach ($path in $agentFiles) {
    $content = Get-Content -Raw -LiteralPath $path
    if ($content -cmatch "Deepseek/") {
        throw "Mixed-case Deepseek model reference remains in $path"
    }
}
$checkConfig = Get-Content -Raw -LiteralPath $openCodexConfig | ConvertFrom-Json
if ($null -ne $checkConfig.providers.deepseek.PSObject.Properties["modelSupportsReasoningSummaries"]) {
    throw "The stale modelSupportsReasoningSummaries override remains in $openCodexConfig"
}
Write-Output "Local OpenCodex configuration updated."

if (-not $SkipSync) {
    & ocx restart
    if ($LASTEXITCODE -ne 0) { throw "ocx restart failed with exit code $LASTEXITCODE" }
    & ocx sync --restart-codex
    if ($LASTEXITCODE -ne 0) { throw "ocx sync failed with exit code $LASTEXITCODE" }
    & ocx ready --wait
    if ($LASTEXITCODE -ne 0) { throw "ocx ready failed with exit code $LASTEXITCODE" }
}

foreach ($path in $catalogFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Write-Warning "Generated catalog was not found: $path"
        continue
    }
    $catalogText = Get-Content -Raw -LiteralPath $path
    if ($catalogText -notmatch '"slug"\s*:\s*"deepseek/deepseek-v4-flash"') {
        Write-Warning "deepseek-v4-flash was not found in $path"
    }
    if ($catalogText -notmatch '"slug"\s*:\s*"deepseek/deepseek-v4-pro"') {
        Write-Warning "deepseek-v4-pro was not found in $path"
    }
}

Write-Output "Done. Reopen Codex and test a Flash agent continuation."
