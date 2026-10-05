param([Parameter(Mandatory=$true)][string]$RunId,
      [string]$PythonPath='D:\RL_JUMP\.venv\python.exe')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$dir=Join-Path $PSScriptRoot "runs\${RunId}_launcher"
if(Test-Path -LiteralPath $dir){throw 'Use a new launcher id'}
New-Item -ItemType Directory -Path $dir|Out-Null
$start=Get-Date
try {
    $arguments=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
      (Join-Path $PSScriptRoot 'run_pipeline_v2.ps1'),'-PythonPath',$PythonPath,'-RunId',$RunId)
    $child=Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput (Join-Path $dir 'stdout.log') -RedirectStandardError (Join-Path $dir 'stderr.log')
    $processHandle=$child.Handle
    @{supervisor_pid=$PID;wrapper_pid=$child.Id;run=$RunId;
      source_git=(git -C (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) rev-parse HEAD);
      frozen_sha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'FROZEN_V2.json') -Algorithm SHA256).Hash.ToLower();
      utc_start=$start.ToUniversalTime().ToString('o')}|ConvertTo-Json|
      Set-Content -LiteralPath (Join-Path $dir 'launcher.json') -Encoding UTF8
    $child.WaitForExit()
    $code=$child.ExitCode
    if($null -eq $code){throw 'Native child exit code unavailable'}
    @{process_exited=$true;exit_code=$code;wrapper_pid=$child.Id;
      utc_start=$start.ToUniversalTime().ToString('o');utc_end=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'wrapper_exit_receipt.json') -Encoding UTF8
    exit $code
} catch {
    @{status='SUPERVISOR_FAILED';error=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'supervisor_failure.json') -Encoding UTF8
    throw
}
