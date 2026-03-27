param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "launch_all.py"

function Resolve-RepoRoot {
    param([string]$StartDir)

    $searchDir = $StartDir
    while ($true) {
        if (Test-Path (Join-Path $searchDir "pyproject.toml")) {
            return $searchDir
        }
        $parent = Split-Path $searchDir -Parent
        if (-not $parent -or $parent -eq $searchDir) {
            break
        }
        $searchDir = $parent
    }

    throw "Could not locate repo root (pyproject.toml) from $StartDir"
}

if (-not (Test-Path $runner)) {
    throw "Launcher not found: $runner"
}

$repoRoot = $null
try {
    $repoRoot = Resolve-RepoRoot -StartDir $scriptDir
}
catch {
    $repoRoot = $null
}
if (-not $repoRoot) {
    $repoRoot = Resolve-RepoRoot -StartDir (Get-Location).Path
}
Push-Location $repoRoot
try {
    & uv run python $runner @RunnerArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
