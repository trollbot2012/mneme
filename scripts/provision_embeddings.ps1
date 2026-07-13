<#
.SYNOPSIS
Provision the ADR-0004 embedding encoder artifacts (all-MiniLM-L6-v2 quint8
avx2 ONNX export + vocab.txt) into a models/ directory.

.DESCRIPTION
CHECK-THEN-SKIP: a file already present with a matching pinned sha256 is never
re-downloaded -- a provisioned box exits 0 with zero network traffic. Downloads
stage to <name>.part and rename atomically; any sha256 mismatch deletes the
downloaded copy and fails loud.

This script is the ONLY place network provisioning happens: mneme.py ships
zero network code, and the optional onnxruntime wheel is NEVER pip-installed
here (see the hint printed at the end).

The URL pins the upstream revision matching ENCODER_ID (rev1110a243); the
pinned sha256 below -- taken from the load-verified deployment artifact -- is
the actual integrity gate for both files.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/provision_embeddings.ps1
powershell -ExecutionPolicy Bypass -File scripts/provision_embeddings.ps1 -Dest C:\path\to\models
#>
[CmdletBinding()]
param(
    # Destination directory (default: $env:MNEME_HOME\models, else ~\.mneme\models)
    [string]$Dest = ""
)

$ErrorActionPreference = "Stop"

if (-not $Dest) {
    $mnemeHome = if ($env:MNEME_HOME) { $env:MNEME_HOME } else { Join-Path $env:USERPROFILE ".mneme" }
    $Dest = Join-Path $mnemeHome "models"
}

# Pinned upstream revision (baked into ENCODER_ID) and per-file sha256 pins.
$Rev = "1110a243"
$BaseUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$Rev"
$Files = @(
    @{ Name = "model_quint8_avx2.onnx"; Url = "$BaseUrl/onnx/model_quint8_avx2.onnx";
       Bytes = 23046789; Sha256 = "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21" },
    @{ Name = "vocab.txt"; Url = "$BaseUrl/vocab.txt";
       Bytes = 231508; Sha256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3" }
)

function Get-Sha256Hex([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$downloaded = 0
foreach ($f in $Files) {
    $target = Join-Path $Dest $f.Name
    if (Test-Path -LiteralPath $target) {
        if ((Get-Sha256Hex $target) -eq $f.Sha256) {
            Write-Host "OK (already provisioned, sha256 verified): $target"
            continue   # check-then-skip: no network for present, verified files
        }
        Write-Warning "sha256 mismatch on existing $($f.Name) -- deleting and refetching"
        Remove-Item -LiteralPath $target -Force
    }
    $part = "$target.part"
    if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    Write-Host "downloading $($f.Url) -> $target ($($f.Bytes) bytes)"
    # PS 5.1 defaults can exclude TLS 1.2; huggingface requires it.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $f.Url -OutFile $part -UseBasicParsing
    $got = Get-Sha256Hex $part
    if ($got -ne $f.Sha256) {
        Remove-Item -LiteralPath $part -Force   # delete on mismatch, fail loud
        throw "sha256 mismatch for $($f.Name): expected $($f.Sha256), got $got -- downloaded copy deleted"
    }
    Move-Item -LiteralPath $part -Destination $target -Force   # atomic-rename publish
    Write-Host "OK (downloaded, sha256 verified): $target"
    $downloaded++
}

Write-Host ""
Write-Host "models ready in: $Dest"
Write-Host "Embeddings also need the optional onnxruntime wheel (NOT installed by this script):"
Write-Host '    pip install "onnxruntime>=1.20"        # or: pip install "mneme-memory[embeddings]"'
Write-Host "Point Mneme at the directory via the 'embed_model_dir' config key,"
Write-Host "or place models/ next to the store's mneme.db -- 'auto' does the rest."
