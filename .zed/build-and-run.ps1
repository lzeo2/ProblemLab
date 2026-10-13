$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path '.uv-cache'
$ortRoot = Join-Path (Get-Location).Path 'third_party\onnxruntime-win-x64-1.30.0\onnxruntime-win-x64-1.30.0'
$ortInclude = Join-Path $ortRoot 'include'
$ortLib = Join-Path $ortRoot 'lib'
$ortDll = Join-Path $ortLib 'onnxruntime.dll'

& g++ -std=c++17 -Wall -Wextra -Wpedantic -I third_party -I $ortInclude main.cpp (Join-Path $ortLib 'onnxruntime.lib') -o problemlab.exe
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Copy-Item -LiteralPath $ortDll -Destination (Get-Location).Path -Force
Write-Output 'Build succeeded. Running ProblemLab...'
& .\problemlab.exe
exit $LASTEXITCODE
