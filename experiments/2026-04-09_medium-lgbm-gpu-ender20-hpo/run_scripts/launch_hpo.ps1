$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$experimentDir = Split-Path -Parent $scriptDir
$repoRoot = Split-Path -Parent (Split-Path -Parent $experimentDir)
$launchStatePath = Join-Path $scriptDir "hpo_launch.json"
$studyConfigPath = Join-Path $experimentDir "configs\hpo_study_v1.json"
$numerengExe = Join-Path $repoRoot ".venv\Scripts\numereng.exe"

if (-not (Test-Path $numerengExe)) {
    throw "numereng_executable_missing:$numerengExe"
}

if (-not (Test-Path $studyConfigPath)) {
    throw "hpo_study_config_missing:$studyConfigPath"
}

if (Test-Path $launchStatePath) {
    $existing = Get-Content $launchStatePath -Raw | ConvertFrom-Json
    if ($null -ne $existing.pid) {
        try {
            $process = Get-Process -Id ([int]$existing.pid) -ErrorAction Stop
            $state = [ordered]@{
                already_running = $true
                pid = $process.Id
                stdout_path = $existing.stdout_path
                stderr_path = $existing.stderr_path
                launched_at = $existing.launched_at
            }
            $state | ConvertTo-Json -Depth 4
            exit 0
        }
        catch {
        }
    }
}

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$stdoutPath = Join-Path $scriptDir "hpo_stdout_$stamp.log"
$stderrPath = Join-Path $scriptDir "hpo_stderr_$stamp.log"
$arguments = @(
    "hpo",
    "create",
    "--study-config",
    $studyConfigPath,
    "--workspace",
    $repoRoot
)

$process = Start-Process `
    -FilePath $numerengExe `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$payload = [ordered]@{
    pid = $process.Id
    stdout_path = $stdoutPath
    stderr_path = $stderrPath
    study_config_path = $studyConfigPath
    launched_at = (Get-Date).ToString("o")
}
$payload | ConvertTo-Json -Depth 4 | Set-Content -Path $launchStatePath -Encoding utf8
$payload | ConvertTo-Json -Depth 4
