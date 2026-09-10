# Deploy app11.nextaura.us to IBM Code Engine (CPU-only, no GPU).
param(
    [string]$CeProject = "nextaura-workflows",
    [string]$CeRegion = "us-south",
    [string]$AppName = "app11-nextaura-us",
    [string]$AppRoot = "E:\Under32testing\app11-nextaura-us-github",
    [string]$PublicUrl = "https://app11.nextaura.us"
)

$ErrorActionPreference = "Stop"
Write-Host "=== Deploy $AppName (open-source CPU training lab) ===" -ForegroundColor Cyan

ibmcloud target -g Default | Out-Null
ibmcloud target -r $CeRegion | Out-Null
ibmcloud ce project select -n $CeProject | Out-Null

$exists = $false
try {
    ibmcloud ce app get -n $AppName 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $exists = $true }
} catch {}

$ceArgs = @(
    "--name", $AppName,
    "--build-source", $AppRoot,
    "--build-dockerfile", "Dockerfile",
    "--build-strategy", "dockerfile",
    "--cpu", "2",
    "--memory", "4G",
    "--ephemeral-storage", "1G",
    "--port", "8080",
    "--min-scale", "1",
    "--max-scale", "1",
    "--timeout", "300",
    "--build-timeout", "2400",
    "-e", "PUBLIC_BASE_URL=$PublicUrl",
    "-e", "PUBLIC_HOST=app11.nextaura.us",
    "-e", "PUBLIC_URL=$PublicUrl",
    "-e", "TRAIN_DEVICE=cpu",
    "-e", "HEADLESS=1"
)

if ($exists) {
    ibmcloud ce app update @ceArgs --rebuild
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Rebuild failed - retry without --rebuild" -ForegroundColor Yellow
        ibmcloud ce app update @ceArgs
    }
} else {
    ibmcloud ce app create @ceArgs
}
if ($LASTEXITCODE -ne 0) { throw "CE deploy failed" }

$ceUrl = (ibmcloud ce app get -n $AppName -o json | ConvertFrom-Json).status.url
Write-Host "CE URL: $ceUrl" -ForegroundColor Green

$workerPath = Join-Path $AppRoot "worker.js"
$content = Get-Content $workerPath -Raw
$content = $content -replace '^const ORIGIN = .*;', "const ORIGIN = `"$ceUrl`";"
Set-Content -Path $workerPath -Value $content -NoNewline

Write-Host "==> Deploy Cloudflare worker nextaura-app11-us" -ForegroundColor Cyan
Remove-Item Env:CLOUDFLARE_API_TOKEN -ErrorAction SilentlyContinue
$sm = "77a74a8e-30d4-440a-b4da-7eb56ff43425"
$cfToken = (ibmcloud secrets-manager secret-by-name --instance-id $sm --region us-south --secret-type arbitrary --name cloudflare-bootstrap-token --secret-group-name github-bootstrap --output json | ConvertFrom-Json).payload
if ($cfToken) { $env:CLOUDFLARE_API_TOKEN = $cfToken }
Push-Location $AppRoot
npx --yes wrangler deploy 2>&1
if ($LASTEXITCODE -ne 0 -and $cfToken) {
    Remove-Item Env:CLOUDFLARE_API_TOKEN -ErrorAction SilentlyContinue
    npx --yes wrangler deploy 2>&1
}
Pop-Location
Remove-Item Env:CLOUDFLARE_API_TOKEN -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) { throw "Cloudflare worker deploy failed" }

Write-Host ""
Write-Host "DONE $PublicUrl -> $ceUrl" -ForegroundColor Green
