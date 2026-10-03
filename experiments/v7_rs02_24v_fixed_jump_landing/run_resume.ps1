param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [Parameter(Mandatory=$true)][string]$SourceRun,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][int]$ExpectedUpdate,
    [Parameter(Mandatory=$true)][string]$RunId
)
$ErrorActionPreference='Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid RunId' }
$PythonPath=(Resolve-Path -LiteralPath $PythonPath).Path
$trainDir=Join-Path $PSScriptRoot "runs\$RunId"
if (Test-Path -LiteralPath $trainDir) { throw 'Use a fresh RunId' }
$evalId="${RunId}_eval"
$evalDir=Join-Path $PSScriptRoot "runs\$evalId"
if (Test-Path -LiteralPath $evalDir) { throw 'Use a fresh evaluation id' }
$frozenHash=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'FROZEN.json') -Algorithm SHA256).Hash.ToLowerInvariant()
$resumeHash=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'RESUME_FROZEN.json') -Algorithm SHA256).Hash.ToLowerInvariant()
& $PythonPath (Join-Path $PSScriptRoot 'verify_fast.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonPath (Join-Path $PSScriptRoot 'resume_train.py') --source $SourceRun --checkpoint $Checkpoint --expected-update $ExpectedUpdate --run-id $RunId
$trainExit=$LASTEXITCODE
New-Item -ItemType Directory -Path $trainDir -Force | Out-Null
@{stage='train';run_id=$RunId;process_exited=$true;exit_code=$trainExit;frozen_sha256=$frozenHash;
  resume_frozen_sha256=$resumeHash;utc_end=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $trainDir 'exit_receipt.json') -Encoding UTF8
if ($trainExit -ne 0) { exit $trainExit }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_checked.ps1') -Stage evaluate -RunId $evalId -PythonPath $PythonPath -TrainResult (Join-Path $trainDir 'result.json')
$evalExit=$LASTEXITCODE
if ($evalExit -ne 0) { exit $evalExit }
& $PythonPath (Join-Path $PSScriptRoot 'audit.py') --training $trainDir --evaluation $evalDir
$auditExit=$LASTEXITCODE
if ($auditExit -eq 0) {
    & $PythonPath (Join-Path $PSScriptRoot 'resume_audit.py') --training $trainDir --evaluation $evalDir
    $auditExit=$LASTEXITCODE
}
@{status=$(if($auditExit -eq 0){'COMPLETED'}else{'AUDIT_FAILED'});train_exit=$trainExit;
  eval_exit=$evalExit;audit_exit=$auditExit;resume_frozen_sha256=$resumeHash;utc_end=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evalDir 'pipeline_receipt.json') -Encoding UTF8
exit $auditExit
