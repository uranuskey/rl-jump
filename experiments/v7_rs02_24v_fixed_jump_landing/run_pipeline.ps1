param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [Parameter(Mandatory=$true)][string]$SmokeResult,
    [Parameter(Mandatory=$true)][string]$RunId,
    [ValidateSet(256,2048)][int]$NumEnvs=2048
)
$ErrorActionPreference='Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid RunId' }
$wrapper=Join-Path $PSScriptRoot 'run_checked.ps1'
$trainDir=Join-Path $PSScriptRoot "runs\$RunId"
$evalId="${RunId}_eval"
$evalDir=Join-Path $PSScriptRoot "runs\$evalId"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper -Stage train -RunId $RunId -PythonPath $PythonPath -NumEnvs $NumEnvs -SmokeResult $SmokeResult
$trainExit=$LASTEXITCODE
if ($trainExit -ne 0) { exit $trainExit }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper -Stage evaluate -RunId $evalId -PythonPath $PythonPath -TrainResult (Join-Path $trainDir 'result.json')
$evalExit=$LASTEXITCODE
if ($evalExit -ne 0) { exit $evalExit }
& $PythonPath (Join-Path $PSScriptRoot 'audit.py') --training $trainDir --evaluation $evalDir
$auditExit=$LASTEXITCODE
@{status=$(if($auditExit -eq 0){'COMPLETED'}else{'AUDIT_FAILED'});train_exit=$trainExit;eval_exit=$evalExit;audit_exit=$auditExit;utc_end=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evalDir 'pipeline_receipt.json') -Encoding UTF8
exit $auditExit
