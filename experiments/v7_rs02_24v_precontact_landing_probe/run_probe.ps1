param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$Checkpoint,
      [Parameter(Mandatory=$true)][string]$RunId,
      [switch]$ProtectLanding,
      [switch]$Coupled)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
if($Coupled -and -not $ProtectLanding){throw 'Coupled trial requires ProtectLanding'}
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
    if($ProtectLanding){$Extra+=@('--protect-landing')}
    if($Coupled){$Extra+=@('--coupled')}
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot 'precontact_probe.py') --checkpoint $Checkpoint --run-id $name @Extra
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
    if($ProtectLanding){
        Run-Stage 'baseline_repeat' @('--mode','native')
        $baseRepeat=Get-Content -LiteralPath (Join-Path $runs "${RunId}_baseline_repeat\result.json") -Raw|ConvertFrom-Json
        if($baseRepeat.all_metrics.passed -ne 45){Save-Pipeline 'BASELINE_NOT_QUALIFIED' 'baseline_repeat';exit 2}
    }
    Run-Stage 'smoke' @('--mode','native','--profile','1','--baseline',$baseline)
    Run-Stage 'search' @('--mode','search')
    $search=Join-Path $runs "${RunId}_search\result.json"
    Run-Stage 'candidate0' @('--mode','native','--search',$search,'--rank','0','--baseline',$baseline)
    Run-Stage 'candidate1' @('--mode','native','--search',$search,'--rank','1','--baseline',$baseline)
    $candidates=@()
    foreach($rank in @(0,1)){
        $path=Join-Path $runs "${RunId}_candidate$rank\result.json"
        $r=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
        if($r.status -eq 'EVALUATED' -and $r.admitted_vs_native_baseline){
            $candidates+=@{rank=$rank;mean_peak_n=$r.all_metrics.mean_force_n;profile=$r.profiles[0];result=$path}
        }
    }
    if($candidates.Count -eq 0){
        @{status='NO_NATIVE_CANDIDATE';chosen=$null;training_updates=0}|ConvertTo-Json|
          Set-Content -LiteralPath (Join-Path $resultDir 'selection.json') -Encoding UTF8
        Save-Pipeline 'COMPLETED_WITHOUT_CANDIDATE' 'native_selection'
        exit 0
    }
    $winner=$candidates|Sort-Object mean_peak_n|Select-Object -First 1
    $winner|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $resultDir 'selection.json') -Encoding UTF8
    Run-Stage 'repeat' @('--mode','native','--search',$search,'--rank',([string]$winner.rank),'--baseline',$baseline)
    $repeat=Get-Content -LiteralPath (Join-Path $runs "${RunId}_repeat\result.json") -Raw|ConvertFrom-Json
    if(-not $repeat.admitted_vs_native_baseline){Save-Pipeline 'CANDIDATE_NOT_REPRODUCED' 'repeat';exit 0}
    Run-Stage 'raised' @('--mode','native','--search',$search,'--rank',([string]$winner.rank),'--baseline',$baseline,'--height-m','0.01')
    Save-Pipeline 'COMPLETED' 'raised'
} catch {
    @{status='WRAPPER_FAILED';error=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $resultDir 'wrapper_failure.json') -Encoding UTF8
    Save-Pipeline 'WRAPPER_FAILED' 'wrapper'
    throw
}
