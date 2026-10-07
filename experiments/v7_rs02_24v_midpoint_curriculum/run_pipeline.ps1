param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$runs=Join-Path $PSScriptRoot 'runs'
$dir=Join-Path $runs $RunId
$pipeline=Join-Path $runs "${RunId}_pipeline"
if((Test-Path -LiteralPath $dir) -or (Test-Path -LiteralPath $pipeline)){throw 'Use a new run id'}
New-Item -ItemType Directory -Path $pipeline|Out-Null
$script:stages=@()
function Save-Pipeline([string]$Status,[string]$Stage){
    @{status=$Status;stage=$Stage;stages=$script:stages;utc=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $pipeline 'pipeline_receipt.json') -Encoding UTF8
}
function Run-Stage([string]$Name,[string]$Folder,[string]$Script,[string[]]$Extra){
    $dir=Join-Path $runs $Folder
    Save-Pipeline 'RUNNING' $Name
    $start=Get-Date
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot $Script) @Extra
    $code=$LASTEXITCODE
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    $receipt=@{stage=$Name;process_exited=$true;exit_code=$code;
      utc_start=$start.ToUniversalTime().ToString('o');utc_end=[DateTime]::UtcNow.ToString('o')}
    $file=if($Name -eq 'audit'){'audit_exit_receipt.json'}else{'exit_receipt.json'}
    $receipt|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir $file) -Encoding UTF8
    $script:stages+=$receipt
    if($code -ne 0){
        $events=@(Get-WinEvent -FilterHashtable @{LogName='Application';StartTime=$start} -ErrorAction SilentlyContinue|
          Where-Object {$_.Id -in @(1000,1001) -and $_.Message -match 'python|c10_cuda|torch'}|
          Select-Object -First 5 TimeCreated,Id,Message)
        @{stage=$Name;exit_code=$code;events=$events;root_cause_resolved=$false}|
          ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $dir 'runtime_failure.json') -Encoding UTF8
        Save-Pipeline 'FAILED' $Name
        exit $code
    }
}
try {
    Run-Stage 'curriculum' $RunId 'midpoint_course.py' @('--run-id',$RunId)
    Run-Stage 'audit' $RunId 'midpoint_audit.py' @('--run',(Join-Path $runs $RunId))
    Save-Pipeline 'COMPLETED' 'audit'
} catch {
    @{status='WRAPPER_FAILED';error=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $pipeline 'wrapper_failure.json') -Encoding UTF8
    Save-Pipeline 'WRAPPER_FAILED' 'wrapper'
    throw
}
