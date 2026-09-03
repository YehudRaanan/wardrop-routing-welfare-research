[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "build")
)

$ErrorActionPreference = "Stop"
$latexmk = (Get-Command latexmk.exe -ErrorAction Stop).Source

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
Push-Location $PSScriptRoot
try {
    & $latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "-outdir=$OutputDirectory" main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "PDF compilation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

$builtPdf = Join-Path $OutputDirectory "main.pdf"
$publicPdf = Join-Path $PSScriptRoot "evaluating-routing-rules-using-social-welfare.pdf"
if (-not (Test-Path -LiteralPath $builtPdf)) {
    throw "Compilation finished without producing $builtPdf."
}
Copy-Item -LiteralPath $builtPdf -Destination $publicPdf -Force
Write-Output "PDF created at $publicPdf"
