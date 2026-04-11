param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "launch_all.py"

if (-not (Test-Path $runner)) {
    throw "Launcher not found: $runner"
}

& uv run python $runner @RunnerArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
