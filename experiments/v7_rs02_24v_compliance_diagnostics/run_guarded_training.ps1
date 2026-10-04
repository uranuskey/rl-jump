param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$RunId,
      [Parameter(Mandatory=$true)][string]$RuntimeProbe)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid RunId'}
$task=Join-Path (Split-Path $PSScriptRoot -Parent) 'v7_rs02_24v_compliant_landing_v3'
$runs=Join-Path $task 'runs'
$frozen=(Get-FileHash (Join-Path $task 'FROZEN.json') -Algorithm SHA256).Hash.ToLowerInvariant()
$probe=Get-Content -Raw (Join-Path $RuntimeProbe 'result.json')|ConvertFrom-Json
$receipt=Get-Content -Raw (Join-Path $RuntimeProbe 'exit_receipt.json')|ConvertFrom-Json
if($probe.status -ne 'PROBE_PASSED' -or $probe.repeats -ne 3 -or $probe.frozen_sha256 -ne $frozen -or
   !$receipt.process_exited -or $receipt.exit_code -ne 0){throw 'Three-repeat runtime admission required'}
foreach($name in @($RunId,"${RunId}_eval")){
    if(Test-Path (Join-Path $runs $name)){throw 'Use fresh run IDs'}
}
function Invoke-Guarded([string]$Stage,[string]$Name,[string]$Entry,[string[]]$NativeArgs){
    & $PythonPath -X faulthandler -u (Join-Path $task $Entry) @NativeArgs
    $stageExit=$LASTEXITCODE
    $out=Join-Path $runs $Name
    New-Item -ItemType Directory -Path $out -Force|Out-Null
    @{stage=$Stage;run_id=$Name;process_exited=$true;exit_code=$stageExit;frozen_sha256=$frozen;
      utc_end=[DateTime]::UtcNow.ToString('o');launcher='run_guarded_training.ps1'} |
        ConvertTo-Json|Set-Content (Join-Path $out 'exit_receipt.json') -Encoding UTF8
    if($stageExit -ne 0){
        @{status='PROCESS_FAILED';native_crash=($stageExit -lt 0);exit_code=$stageExit;
          progress_file_may_be_stale=$true;utc_end=[DateTime]::UtcNow.ToString('o')} |
          ConvertTo-Json|Set-Content (Join-Path $out 'runtime_failure.json') -Encoding UTF8
        exit $stageExit
    }
}
$validation=Join-Path $runs 'train_01_validate\result.json'
$smoke=Join-Path $runs 'train_01_smoke\result.json'
Invoke-Guarded 'train' $RunId 'train.py' @('--mode','train','--num-envs','512','--run-id',$RunId,'--validation',$validation,'--smoke-result',$smoke)
$trainDir=Join-Path $runs $RunId
$evalId="${RunId}_eval"
$evalDir=Join-Path $runs $evalId
Invoke-Guarded 'evaluate' $evalId 'evaluate.py' @('--training',$trainDir,'--run-id',$evalId)
& $PythonPath -X faulthandler -u (Join-Path $task 'audit.py') --training $trainDir --evaluation $evalDir
$code=$LASTEXITCODE
@{status=$(if($code -eq 0){'COMPLETED'}else{'AUDIT_FAILED'});train_exit=0;eval_exit=0;audit_exit=$code;
  frozen_sha256=$frozen;runtime_probe=$RuntimeProbe;utc_end=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json|Set-Content (Join-Path $evalDir 'pipeline_receipt.json') -Encoding UTF8
exit $code
