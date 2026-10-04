param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$RunId,
      [Parameter(Mandatory=$true)][string]$Validation)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid RunId'}
$out=Join-Path $PSScriptRoot "runs\$RunId"
if(Test-Path $out){throw 'Use a fresh run ID'}
& $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot 'runtime_probe.py') --run-id $RunId --validation $Validation --repeats 3
$code=$LASTEXITCODE
New-Item -ItemType Directory -Path $out -Force|Out-Null
@{stage='runtime_probe';run_id=$RunId;process_exited=$true;exit_code=$code;utc_end=[DateTime]::UtcNow.ToString('o')} |
  ConvertTo-Json|Set-Content (Join-Path $out 'exit_receipt.json') -Encoding UTF8
exit $code
