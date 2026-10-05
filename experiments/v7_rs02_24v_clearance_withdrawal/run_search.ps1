param([string]$RunId='search_01',[string]$PythonPath='D:\RL_JUMP\.venv\python.exe')
$ErrorActionPreference='Stop'
$dir=Join-Path $PSScriptRoot "runs/${RunId}_launcher"
if(Test-Path -LiteralPath $dir){throw 'New run id required'}
New-Item -ItemType Directory -Path $dir|Out-Null
$start=[DateTime]::UtcNow.ToString('o')
$child=Start-Process -FilePath $PythonPath -ArgumentList @('-X','faulthandler','-u',(Join-Path $PSScriptRoot 'search_profiles.py'),'--run-id',$RunId) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $dir 'stdout.log') -RedirectStandardError (Join-Path $dir 'stderr.log')
$handle=$child.Handle
@{pid=$child.Id;wrapper_pid=$PID;utc_start=$start}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'launcher.json') -Encoding UTF8
$child.WaitForExit();$code=$child.ExitCode
@{process_exited=$true;exit_code=$code;utc_start=$start;utc_end=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'exit_receipt.json') -Encoding UTF8
exit $code
