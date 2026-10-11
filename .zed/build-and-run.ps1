$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path '.uv-cache'

& g++ -std=c++17 -Wall -Wextra -Wpedantic -I third_party main.cpp -o problemlab.exe
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output 'Build succeeded. Running ProblemLab...'
& .\problemlab.exe
exit $LASTEXITCODE
