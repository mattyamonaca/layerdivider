Set-Location $PSScriptRoot

if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "Installing deps..."
pip install --upgrade -r requirements.txt

Write-Output "Installing torch+cuda"
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

Write-Output "Searching segment model..."
$url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

$dest="./segment_model/sam_vit_h_4b8939.pth"

if (!(Test-Path -Path $dest)) {
    Write-Output  "Downloading segment model to ./segment_model/sam_vit_h_4b8939.pth"
    Write-Output  "or you can close now and download it form https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth by yourself"
    $client = New-Object System.Net.WebClient
    $client.DownloadFile($url, $dest)
}

Write-Output "Install completed"
Read-Host | Out-Null ;