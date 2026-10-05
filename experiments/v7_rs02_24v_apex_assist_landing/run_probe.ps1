param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid run id'}
$dir=Join-Path $PSScriptRoot "runs\$RunId"
if(Test-Path -LiteralPath $dir){throw 'Use a new run id'}
$start=Get-Date
$code=1
try {
    & $PythonPath -X faulthandler -u (Join-Path $PSScriptRoot 'apex_probe.py') --run-id $RunId
    $code=$LASTEXITCODE
} catch {
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    @{status='WRAPPER_FAILED';error=$_.Exception.Message} | ConvertTo-Json |
      Set-Content -LiteralPath (Join-Path $dir 'wrapper_failure.json') -Encoding UTF8
} finally {
    New-Item -ItemType Directory -Path $dir -Force|Out-Null
    @{stage='probe';process_exited=$true;exit_code=$code;
      utc_start=$start.ToUniversalTime().ToString('o');utc_end=[DateTime]::UtcNow.ToString('o')} |
      ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dir 'exit_receipt.json') -Encoding UTF8
    if($code -ne 0){
        $events=@(Get-WinEvent -FilterHashtable @{LogName='Application';StartTime=$start} -ErrorAction SilentlyContinue |
          Where-Object {$_.Id -in @(1000,1001) -and $_.Message -match 'python|c10_cuda|torch'} |
          Select-Object -First 5 TimeCreated,Id,Message)
        @{exit_code=$code;events=$events;root_cause_resolved=$false} | ConvertTo-Json -Depth 6 |
          Set-Content -LiteralPath (Join-Path $dir 'runtime_failure.json') -Encoding UTF8
    }
}
exit $code
