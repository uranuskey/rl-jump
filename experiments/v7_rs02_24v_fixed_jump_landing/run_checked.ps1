param(
    [Parameter(Mandatory=$true)][ValidateSet('smoke','train','evaluate')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [ValidateSet(45,256,2048)][int]$NumEnvs=45,
    [string]$SmokeResult,
    [string]$TrainResult
)
$ErrorActionPreference='Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid RunId' }
$PythonPath=(Resolve-Path -LiteralPath $PythonPath).Path
$runPath=Join-Path $PSScriptRoot "runs\$RunId"
if (Test-Path -LiteralPath $runPath) { throw 'Use a fresh RunId' }
$frozenHash=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'FROZEN.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Stage -eq 'evaluate') {
    if (-not $TrainResult) { throw 'Evaluate requires -TrainResult' }
    $entryPath=Join-Path $PSScriptRoot 'evaluate.py'
    $entryArgs=@('--train-result',$TrainResult,'--run-id',$RunId)
} else {
    if ($Stage -eq 'train' -and -not $SmokeResult) { throw 'Train requires -SmokeResult' }
    $entryPath=Join-Path $PSScriptRoot 'train.py'
    $entryArgs=@('--mode',$Stage,'--num-envs',"$NumEnvs",'--run-id',$RunId)
    if ($Stage -eq 'train') { $entryArgs+=@('--smoke-result',$SmokeResult) }
}
& $PythonPath $entryPath @entryArgs
$processExitCode=$LASTEXITCODE
if ($null -eq $processExitCode) { throw 'Missing native process exit code' }
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
@{stage=$Stage;run_id=$RunId;process_exited=$true;exit_code=$processExitCode;
  frozen_sha256=$frozenHash;utc_end=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runPath 'exit_receipt.json') -Encoding UTF8
exit $processExitCode
