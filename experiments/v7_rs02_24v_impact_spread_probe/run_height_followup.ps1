# Continue native diagnostics after probe_02; retain its completed screening.
param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$Checkpoint,
      [Parameter(Mandatory=$true)][string]$Baseline,
      [Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$base=Get-Content -LiteralPath $Baseline -Raw|ConvertFrom-Json
$receipt=Get-Content -LiteralPath (Join-Path (Split-Path $Baseline) 'exit_receipt.json') -Raw|ConvertFrom-Json
if($base.status -ne 'EVALUATED' -or $base.all_metrics.passed -ne 45 -or $receipt.exit_code -ne 0){throw 'Baseline not verified'}
$runs=Join-Path $PSScriptRoot 'runs'
$resultDir=Join-Path $runs $RunId
if(Test-Path -LiteralPath $resultDir){throw 'Use a new run id'}
New-Item -ItemType Directory -Path $resultDir|Out-Null
function Run-Stage([string]$Suffix,[string[]]$Extra){
    $name="${RunId}_${Suffix}"
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot 'spread_probe.py') --checkpoint $Checkpoint --run-id $name --mode native --baseline $Baseline @Extra
    $code=$LASTEXITCODE
    $dir=Join-Path $runs $name
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    @{process_exited=$true;exit_code=$code;utc_end=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'exit_receipt.json') -Encoding UTF8
    if($code -ne 0){
        @{status='FAILED';stage=$Suffix;exit_code=$code;utc_end=[DateTime]::UtcNow.ToString('o')}|
          ConvertTo-Json|Set-Content -LiteralPath (Join-Path $resultDir 'pipeline_receipt.json') -Encoding UTF8
        exit $code
    }
}
Run-Stage 'raised' @('--height-m','0.01')
# air_minus5mm narrowly missed the predeclared 3% search threshold. These are
# independent diagnostics; do not relabel the fast search as an admitted result.
Run-Stage 'near_miss_flat' @('--profile','1')
Run-Stage 'near_miss_raised' @('--profile','1','--height-m','0.01')
@{status='COMPLETED';all_process_exit_codes=0;training_updates=0;fast_search_admitted=$false;utc_end=[DateTime]::UtcNow.ToString('o')}|
  ConvertTo-Json|Set-Content -LiteralPath (Join-Path $resultDir 'pipeline_receipt.json') -Encoding UTF8
