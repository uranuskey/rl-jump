param(
    [Parameter(Mandatory=$true)][ValidateSet('smoke','train','evaluate')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$RunId,
    [ValidateSet(45,256,2048)][int]$NumEnvs=45,
    [string]$SmokeResult,
    [string]$TrainResult
)
$ErrorActionPreference='Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunId must contain only letters, numbers, underscores and hyphens' }
$pythonPath=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the checkout .venv first' }
$taskPath=Join-Path $PSScriptRoot 'experiments\v7_rs02_24v_flat_landing_reward'
$runPath=Join-Path $taskPath "runs\$RunId"
if (Test-Path -LiteralPath $runPath) { throw 'Use a fresh RunId; this output directory already exists' }
$frozenHash=(Get-FileHash -LiteralPath (Join-Path $taskPath 'FROZEN.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Stage -eq 'evaluate') {
    if (-not $TrainResult) { throw 'Evaluate requires -TrainResult' }
    $entryPath=Join-Path $taskPath 'evaluate_training.py'
    $entryArgs=@('--train-result',$TrainResult,'--run-id',$RunId)
} else {
    if ($Stage -eq 'train' -and -not $SmokeResult) { throw 'Train requires -SmokeResult' }
    $entryPath=Join-Path $taskPath 'train.py'
    $entryArgs=@('--mode',$Stage,'--num-envs',"$NumEnvs",'--run-id',$RunId)
    if ($Stage -eq 'train') { $entryArgs+=@('--smoke-result',$SmokeResult) }
}
# A synchronous native invocation returns only after the Python process exits.
& $pythonPath $entryPath @entryArgs
$processExitCode=$LASTEXITCODE
if ($null -eq $processExitCode) { throw 'Native process returned no exit code' }
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
@{
    stage=$Stage; run_id=$RunId; process_exited=$true; exit_code=$processExitCode
    frozen_sha256=$frozenHash; utc_end=[DateTime]::UtcNow.ToString('o')
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runPath 'exit_receipt.json') -Encoding UTF8
exit $processExitCode
