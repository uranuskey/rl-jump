# Independent native checks after the mixed search admitted no candidate.
param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$Checkpoint,
      [Parameter(Mandatory=$true)][string]$Baseline,
      [Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$runs=Join-Path $PSScriptRoot 'runs'
$root=Join-Path $runs $RunId
if(Test-Path -LiteralPath $root){throw 'Use a new run id'}
New-Item -ItemType Directory -Path $root|Out-Null
foreach($item in @(@('baseline',0),@('soft85',1))){
    $name=$RunId+'_'+$item[0]
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot 'staged_probe.py') --mode native --run-id $name --profile $item[1] --checkpoint $Checkpoint --baseline $Baseline
    $code=$LASTEXITCODE
    $dir=Join-Path $runs $name
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    @{stage=$item[0];process_exited=$true;exit_code=$code;utc_end=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'exit_receipt.json') -Encoding UTF8
    if($code -ne 0){
        @{status='FAILED';stage=$item[0];exit_code=$code}|ConvertTo-Json|
          Set-Content -LiteralPath (Join-Path $root 'pipeline_receipt.json') -Encoding UTF8
        exit $code
    }
}
@{status='DIAGNOSTIC_COMPLETE';all_process_exit_codes=0;training_updates=0;utc_end=[DateTime]::UtcNow.ToString('o')}|
  ConvertTo-Json|Set-Content -LiteralPath (Join-Path $root 'pipeline_receipt.json') -Encoding UTF8
