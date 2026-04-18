param(
    [string]$SimExePath = "D:\course_assignments\in_doing\Comprehensive_Project\Windows\CityAirSim.exe",
    [switch]$ForceRestart,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# 1) Basic checks
if (-not (Test-Path -LiteralPath $SimExePath)) {
    throw "Simulator executable not found: $SimExePath"
}

$settingsPath = Join-Path $env:USERPROFILE "Documents\AirSim\settings.json"
if (-not (Test-Path -LiteralPath $settingsPath)) {
    throw "AirSim settings file not found: $settingsPath"
}

# 2) Minimal config change for headless training: ViewMode=NoDisplay
$settingsObj = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
if ($settingsObj.ViewMode -ne "NoDisplay") {
    $backupPath = "$settingsPath.bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item -LiteralPath $settingsPath -Destination $backupPath -Force
    $settingsObj.ViewMode = "NoDisplay"
    $settingsObj | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $settingsPath -Encoding UTF8
    Write-Host "Updated ViewMode=NoDisplay, backup: $backupPath"
}
else {
    Write-Host "ViewMode is already NoDisplay."
}

# 3) Optional restart: stop old simulator process first
if ($ForceRestart) {
    $old = Get-Process -Name "CityAirSim" -ErrorAction SilentlyContinue
    if ($old) {
        $old | Stop-Process -Force
        Start-Sleep -Seconds 1
        Write-Host "Stopped existing CityAirSim process."
    }
}

# 4) Headless launch args
# -RenderOffscreen keeps camera rendering without opening a visible window.
# -NoSound/-Unattended/-NoSplash reduce overhead for training.
# -settings explicitly pins the config file, preventing fallback to cwd settings.json.
$simArgs = @(
    ('-settings="' + $settingsPath + '"'),
    "-RenderOffscreen",
    "-NoSound",
    "-Unattended",
    "-NoSplash",
    "-windowed",
    "-ResX=640",
    "-ResY=360"
)

$preview = "$SimExePath " + ($simArgs -join " ")
Write-Host "Launch command: $preview"

if ($DryRun) {
    Write-Host "DryRun mode: command printed only."
    return
}

$proc = Start-Process -FilePath $SimExePath -ArgumentList $simArgs -PassThru
Write-Host "CityAirSim started, PID=$($proc.Id)"
Write-Host "Keep this process alive, then run TD3 training."
