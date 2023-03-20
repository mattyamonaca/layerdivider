Set-Location $PSScriptRoot

if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "Installing deps..."
pip install --upgrade -r requirements.txt

Write-Output "Install completed"
Read-Host | Out-Null ;