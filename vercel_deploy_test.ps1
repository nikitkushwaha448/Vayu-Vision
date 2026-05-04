Set-Location -Path (Split-Path -Path $PSScriptRoot -Parent)

$uvicorn = "$PSScriptRoot\\.venv311\\Scripts\\python.exe"
if (-Not (Test-Path $uvicorn)) {
    Write-Error "Python executable not found at $uvicorn. Activate your venv or adjust the path."
    exit 1
}

$proc = Start-Process -FilePath $uvicorn -ArgumentList "-m uvicorn api.index:app --host 127.0.0.1 --port 8000" -PassThru
Write-Output "Started uvicorn (PID $($proc.Id)), waiting 2s..."
Start-Sleep -Seconds 2

try {
    $h = Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method GET -TimeoutSec 5
    Write-Output "Health: $($h | ConvertTo-Json)"
} catch {
    Write-Error "Health check failed: $_"
}

try {
    $predict = Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method POST -Body (ConvertTo-Json @{city='Delhi'; pollutant_values=@{pm25=50;pm10=40;o3=10;no2=15;so2=2;co=0.2}}) -ContentType 'application/json' -TimeoutSec 5
    Write-Output "Predict: $($predict | ConvertTo-Json)"
} catch {
    Write-Error "Predict call failed: $_"
}

if ($proc -and -not $proc.HasExited) { $proc | Stop-Process }
Write-Output "Done."
