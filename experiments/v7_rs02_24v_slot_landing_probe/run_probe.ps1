param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$Checkpoint,
      [Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$runs=Join-Path $PSScriptRoot 'runs'
$resultDir=Join-Path $runs $RunId
if(Test-Path -LiteralPath $resultDir){throw 'Use a new run id'}
New-Item -ItemType Directory -Path $resultDir|Out-Null
$script:receipts=@()
function Save-Pipeline([string]$Status,[string]$Stage){
    @{status=$Status;stage=$Stage;stages=$script:receipts;training_updates=0;
      utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Depth 8|
      Set-Content -LiteralPath (Join-Path $resultDir 'pipeline_receipt.json') -Encoding UTF8
}
function Run-Stage([string]$Suffix,[string[]]$Extra){
    $name="${RunId}_${Suffix}"
    Save-Pipeline 'RUNNING' $Suffix
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot 'slot_probe.py') --checkpoint $Checkpoint --run-id $name @Extra
    $code=$LASTEXITCODE
    $dir=Join-Path $runs $name
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    $receipt=@{stage=$Suffix;process_exited=$true;exit_code=$code;utc_end=[DateTime]::UtcNow.ToString('o')}
    $receipt|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'exit_receipt.json') -Encoding UTF8
    $script:receipts+=$receipt
    if($code -ne 0){Save-Pipeline 'FAILED' $Suffix;exit $code}
}
try {
    Run-Stage 'baseline' @('--mode','native')
    $baseline=Join-Path $runs "${RunId}_baseline\result.json"
    $base=Get-Content -LiteralPath $baseline -Raw|ConvertFrom-Json
    if($base.all_metrics.passed -ne 45){Save-Pipeline 'BASELINE_NOT_QUALIFIED' 'baseline';exit 2}
    Run-Stage 'slot_only' @('--mode','native','--profile','1','--baseline',$baseline)
    Run-Stage 'search' @('--mode','search')
    $search=Join-Path $runs "${RunId}_search\result.json"
    Run-Stage 'candidate0' @('--mode','native','--search',$search,'--rank','0','--baseline',$baseline)
    Run-Stage 'candidate1' @('--mode','native','--search',$search,'--rank','1','--baseline',$baseline)
    $candidates=@()
    foreach($name in @('slot_only','candidate0','candidate1')){
        $path=Join-Path $runs "${RunId}_${name}\result.json"
        $r=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
        if($r.status -eq 'EVALUATED' -and $r.training_seed_qualified){
            $candidates+=@{name=$name;mean_peak_n=$r.all_metrics.mean_force_n;profile=$r.profiles[0];result=$path;admitted=$r.admitted_vs_native_baseline}
        }
    }
    if($candidates.Count -eq 0){
        Save-Pipeline 'COMPLETED_WITHOUT_SEED' 'native_selection';exit 0
    }
    $winner=$candidates|Sort-Object mean_peak_n|Select-Object -First 1
    $winner|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $resultDir 'selection.json') -Encoding UTF8
    Run-Stage 'repeat' @('--mode','native','--profile-result',$winner.result,'--baseline',$baseline)
    $repeat=Get-Content -LiteralPath (Join-Path $runs "${RunId}_repeat\result.json") -Raw|ConvertFrom-Json
    if(-not $repeat.training_seed_qualified){Save-Pipeline 'SEED_NOT_REPRODUCED' 'repeat';exit 0}
    Save-Pipeline 'SEED_VALIDATED' 'repeat'
} catch {
    @{status='WRAPPER_FAILED';error=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $resultDir 'wrapper_failure.json') -Encoding UTF8
    Save-Pipeline 'WRAPPER_FAILED' 'wrapper';throw
}
