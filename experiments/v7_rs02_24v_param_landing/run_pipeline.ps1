param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [Parameter(Mandatory=$true)][string]$RunId
)
$ErrorActionPreference='Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid RunId' }
$PythonPath=(Resolve-Path -LiteralPath $PythonPath).Path
$smokeId="${RunId}_smoke"
$evalId="${RunId}_eval"
foreach ($name in @($RunId,$smokeId,$evalId)) {
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "runs\$name")) { throw 'Use fresh run IDs' }
}
$frozenHash=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'FROZEN.json') -Algorithm SHA256).Hash.ToLowerInvariant()
function Invoke-Stage([string]$Stage,[string]$Name,[string]$Entry,[string[]]$NativeArgs) {
    & $PythonPath (Join-Path $PSScriptRoot $Entry) @NativeArgs
    $code=$LASTEXITCODE
    $folder=Join-Path $PSScriptRoot "runs\$Name"
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    @{stage=$Stage;run_id=$Name;process_exited=$true;exit_code=$code;frozen_sha256=$frozenHash;utc_end=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'exit_receipt.json') -Encoding UTF8
    if ($code -ne 0) { exit $code }
}
& $PythonPath (Join-Path $PSScriptRoot 'verify_cpu.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Invoke-Stage 'smoke' $smokeId 'train.py' @('--mode','smoke','--num-envs','45','--run-id',$smokeId)
$smokeResult=Join-Path $PSScriptRoot "runs\$smokeId\result.json"
Invoke-Stage 'train' $RunId 'train.py' @('--mode','train','--num-envs','512','--run-id',$RunId,'--smoke-result',$smokeResult)
$trainDir=Join-Path $PSScriptRoot "runs\$RunId"
$evalDir=Join-Path $PSScriptRoot "runs\$evalId"
Invoke-Stage 'evaluate' $evalId 'evaluate.py' @('--training',$trainDir,'--run-id',$evalId)
& $PythonPath (Join-Path $PSScriptRoot 'audit.py') --training $trainDir --evaluation $evalDir
$code=$LASTEXITCODE
@{status=$(if($code -eq 0){'COMPLETED'}else{'AUDIT_FAILED'});train_exit=0;eval_exit=0;audit_exit=$code;
  frozen_sha256=$frozenHash;utc_end=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evalDir 'pipeline_receipt.json') -Encoding UTF8
exit $code
