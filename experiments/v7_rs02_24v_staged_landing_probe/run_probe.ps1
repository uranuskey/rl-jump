param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$Checkpoint,
      [Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$runs=Join-Path $PSScriptRoot 'runs'
$resultDir=Join-Path $runs $RunId
if(Test-Path -LiteralPath $resultDir){throw 'Use a new run id'}
New-Item -ItemType Directory -Path $resultDir|Out-Null
function Run-Stage([string]$Suffix,[string[]]$Extra){
    $name="${RunId}_${Suffix}"
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot 'staged_probe.py') --checkpoint $Checkpoint --run-id $name @Extra
    $code=$LASTEXITCODE
    $dir=Join-Path $runs $name
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    @{stage=$Suffix;process_exited=$true;exit_code=$code;utc_end=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'exit_receipt.json') -Encoding UTF8
    if($code -ne 0){
        @{status='FAILED';stage=$Suffix;exit_code=$code;utc_end=[DateTime]::UtcNow.ToString('o')}|
          ConvertTo-Json|Set-Content -LiteralPath (Join-Path $resultDir 'pipeline_receipt.json') -Encoding UTF8
        exit $code
    }
}
Run-Stage 'baseline' @('--mode','native')
$baseline=Join-Path $runs "${RunId}_baseline\result.json"
Run-Stage 'smoke' @('--mode','native','--profile','2','--baseline',$baseline)
Run-Stage 'search' @('--mode','search')
$search=Join-Path $runs "${RunId}_search\result.json"
Run-Stage 'selected' @('--mode','native','--search',$search,'--baseline',$baseline)
Run-Stage 'repeat' @('--mode','native','--search',$search,'--baseline',$baseline)
Run-Stage 'raised' @('--mode','native','--search',$search,'--baseline',$baseline,'--height-m','0.01')
@{status='COMPLETED';all_process_exit_codes=0;training_updates=0;utc_end=[DateTime]::UtcNow.ToString('o')}|
  ConvertTo-Json|Set-Content -LiteralPath (Join-Path $resultDir 'pipeline_receipt.json') -Encoding UTF8
