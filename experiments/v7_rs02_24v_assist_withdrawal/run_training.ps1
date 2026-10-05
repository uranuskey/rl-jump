param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$ProbeResult,
      [Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$runs=Join-Path $PSScriptRoot 'runs'
$pipelineDir=Join-Path $runs "${RunId}_pipeline"
if(Test-Path -LiteralPath $pipelineDir){throw 'Use a new run id'}
New-Item -ItemType Directory -Path $pipelineDir|Out-Null
$script:receipts=@()
function Save-Pipeline([string]$Status,[string]$Stage){
    @{status=$Status;stage=$Stage;stages=$script:receipts;target_updates=128;num_envs=512;
      utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Depth 8|
      Set-Content -LiteralPath (Join-Path $pipelineDir 'pipeline_receipt.json') -Encoding UTF8
}
function Run-Stage([string]$Stage,[string]$Folder,[string]$Script,[string[]]$Extra){
    Save-Pipeline 'RUNNING' $Stage
    $start=Get-Date
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot $Script) @Extra
    $code=$LASTEXITCODE
    $dir=Join-Path $runs $Folder
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    $receipt=@{stage=$Stage;process_exited=$true;exit_code=$code;
      utc_start=$start.ToUniversalTime().ToString('o');utc_end=[DateTime]::UtcNow.ToString('o')}
    $receiptFile=if($Stage -eq 'audit'){'audit_exit_receipt.json'}else{'exit_receipt.json'}
    $receipt|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir $receiptFile) -Encoding UTF8
    $script:receipts+=$receipt
    if($code -ne 0){
        $events=@(Get-WinEvent -FilterHashtable @{LogName='Application';StartTime=$start} -ErrorAction SilentlyContinue|
          Where-Object {$_.Id -in @(1000,1001) -and $_.Message -match 'python|c10_cuda|torch'}|
          Select-Object -First 5 TimeCreated,Id,Message)
        @{stage=$Stage;exit_code=$code;events=$events;root_cause_resolved=$false;
          utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Depth 6|
          Set-Content -LiteralPath (Join-Path $dir 'runtime_failure.json') -Encoding UTF8
        Save-Pipeline 'FAILED' $Stage
        exit $code
    }
}
try {
    Run-Stage 'smoke' "${RunId}_smoke" 'withdrawal_train.py' @('--mode','smoke','--num-envs','45','--run-id',"${RunId}_smoke",'--probe-result',$ProbeResult)
    Run-Stage 'validate' "${RunId}_validate" 'withdrawal_train.py' @('--mode','validate','--num-envs','512','--run-id',"${RunId}_validate",'--probe-result',$ProbeResult)
    $smoke=Join-Path $runs "${RunId}_smoke\result.json"
    $validation=Join-Path $runs "${RunId}_validate\result.json"
    Run-Stage 'train' $RunId 'withdrawal_train.py' @('--mode','train','--num-envs','512','--run-id',$RunId,'--probe-result',$ProbeResult,'--smoke-result',$smoke,'--validation-result',$validation)
    $training=Join-Path $runs $RunId
    Run-Stage 'evaluate' "${RunId}_eval" 'withdrawal_evaluate.py' @('--training',$training,'--run-id',"${RunId}_eval")
    Run-Stage 'audit' "${RunId}_eval" 'withdrawal_training_audit.py' @('--training',$training,'--evaluation',(Join-Path $runs "${RunId}_eval"))
    Save-Pipeline 'COMPLETED' 'audit'
} catch {
    @{status='WRAPPER_FAILED';error=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $pipelineDir 'wrapper_failure.json') -Encoding UTF8
    Save-Pipeline 'WRAPPER_FAILED' 'wrapper'
    throw
}
