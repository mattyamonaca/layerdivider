
# Change Working Directory
Set-Location $PSScriptRoot

# Check whether Pylauncher is installed
$list = (py --list 2> $null)
if(!$?){
    Write-Error "Pytheon launcher failed."
    Pause
    exit
}

# Check minor version of Python
$requiredMinorVersion = "3\.10"
if(!($list -match $requiredMinorVersion)){
    Write-Error -Message "Python 3.10 is not found. Please install Python 3.10.x later than 3.10.8"
    Pause
    exit
}

# Check micro version of Python
$requiredMicroVersion = "3\.10\.(8|9|([1-7]\d))"
$pyVersion = py -3.10 --version
if(!($pyVersion -match $requiredMicroVersion)){
    Write-Error "Installed Python 3.10 is too old to install layerdivider. Please upgrade your Python 3.10"
    Pause
    exit
}

# Create venv folder
if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    py -3.10 -m venv venv
    if(!$?){
        Pause
        exit
    }
}

.\venv\Scripts\activate
if(!$?){
    Pause
    exit
}

# Install dependent packages
Write-Output "Installing deps..."
pip install --upgrade -r requirements.txt
if(!$?){
    Write-Error "`"pip install --upgrade -r requirements.txt
    `" failed."
    Pause
    exit
}

Write-Output "Install completed."
Pause