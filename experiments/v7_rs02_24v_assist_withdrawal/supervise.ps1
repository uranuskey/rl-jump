param([Parameter(Mandatory=$true)][string]$RunId,
      [string]$PythonPath='D:\RL_JUMP\.venv\python.exe',
      [string]$ProbeResult,
      [switch]$SelfTest)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$dir=Join-Path $PSScriptRoot "runs\${RunId}_launcher"
if(Test-Path -LiteralPath $dir){throw 'Use a new launcher id'}
New-Item -ItemType Directory -Path $dir|Out-Null
$start=Get-Date
try {
    if($SelfTest){
        $arguments=@('-NoProfile','-NonInteractive','-Command','"Start-Sleep -Seconds 12; exit 0"')
    } else {
        if(-not (Test-Path -LiteralPath $ProbeResult)){throw 'Missing admitted probe'}
        $arguments=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
          (Join-Path $PSScriptRoot 'run_training.ps1'),'-PythonPath',$PythonPath,
          '-RunId',$RunId,'-ProbeResult',$ProbeResult)
    }
    $child=Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $dir 'stdout.log') -RedirectStandardError (Join-Path $dir 'stderr.log')
    @{supervisor_pid=$PID;wrapper_pid=$child.Id;run=$RunId;self_test=[bool]$SelfTest;
      utc_start=$start.ToUniversalTime().ToString('o')} | ConvertTo-Json |
      Set-Content -LiteralPath (Join-Path $dir 'launcher.json') -Encoding UTF8
    $child.WaitForExit()
    $code=$child.ExitCode
    @{process_exited=$true;exit_code=$code;wrapper_pid=$child.Id;
      utc_start=$start.ToUniversalTime().ToString('o');utc_end=[DateTime]::UtcNow.ToString('o')} |
      ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dir 'wrapper_exit_receipt.json') -Encoding UTF8
    exit $code
} catch {
    @{status='SUPERVISOR_FAILED';error=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')} |
      ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dir 'supervisor_failure.json') -Encoding UTF8
    throw
}
