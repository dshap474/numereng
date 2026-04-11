$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$experimentDir = Split-Path -Parent $scriptDir
$repoRoot = Split-Path -Parent (Split-Path -Parent $experimentDir)
$studyConfigPath = Join-Path $experimentDir "configs\hpo_study_v1.json"
$numerengExe = Join-Path $repoRoot ".venv\Scripts\numereng.exe"
$transcriptPath = Join-Path $scriptDir "hpo_scheduler_console.log"

if (-not (Test-Path $numerengExe)) {
    throw "numereng_executable_missing:$numerengExe"
}

if (-not (Test-Path $studyConfigPath)) {
    throw "hpo_study_config_missing:$studyConfigPath"
}

Start-Transcript -Path $transcriptPath -Force | Out-Null
try {
    Set-Location $repoRoot
    & $numerengExe hpo create --study-config $studyConfigPath --workspace $repoRoot
}
finally {
    Stop-Transcript | Out-Null
}
