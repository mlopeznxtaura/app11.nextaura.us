# Restore app9.nextaura.us training GUI on IBM GPU (app9-train).
# Stops app10 on port 80 and re-enables app9-nextaura.service.
# app9 code + checkpoints remain at /opt/app9-nextaura-us — only systemd was swapped.
param(
    [string]$InstanceName = "app9-train",
    [string]$SshKey = "$env:USERPROFILE\.ssh\golias_ibm",
    [string]$AppRoot = "E:\Under32testing\upcloudtesting500\app9-nextaura-us",
    [string]$PublicUrl = "https://app9.nextaura.us"
)

$ErrorActionPreference = "Stop"
$remote = "ubuntu@150.239.227.60"

Write-Host "==> Stop app10, start app9 on GPU port 80" -ForegroundColor Cyan
$restore = @'
set -e
sudo systemctl stop app10-nextaura 2>/dev/null || true
sudo systemctl disable app10-nextaura 2>/dev/null || true
sudo systemctl enable app9-nextaura
sudo systemctl restart app9-nextaura
sleep 6
curl -sf http://127.0.0.1/api/health | head -c 500 || curl -sf http://127.0.0.1/ | head -c 200
echo
echo RESTORE_OK
'@
& ssh -i $SshKey $remote $restore
if ($LASTEXITCODE -ne 0) { throw "GPU restore failed" }

$origin = "http://150-239-227-60.sslip.io"
Write-Host "==> Verify $origin" -ForegroundColor Cyan
curl.exe -sf --max-time 20 "$origin/api/health" | Write-Host

$workerPath = Join-Path $AppRoot "worker.js"
$content = Get-Content $workerPath -Raw
$content = $content -replace '^const ORIGIN = .*;', "const ORIGIN = `"$origin`";"
Set-Content -Path $workerPath -Value $content -NoNewline

Write-Host "==> Deploy Cloudflare worker nextaura-app9-us" -ForegroundColor Cyan
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

Write-Host ""
Write-Host "DONE $PublicUrl -> app9 GPU training GUI" -ForegroundColor Green
Write-Host "app10 was stopped on this box. Redeploy app10 to IBM CE if you still need app10.nextaura.us" -ForegroundColor Yellow
