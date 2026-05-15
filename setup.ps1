# setupps1.ps1
# Crea un entorno virtual y instala las dependencias del proyecto.
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-Not (Test-Path ".\venv")) {
    python -m venv .\venv
}

$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Error "No se encontró python.exe en venv. Asegúrate de tener Python instalado y accesible desde tu PATH."
    exit 1
}

& $venvPython -m pip install --upgrade pip setuptools wheel
if (-Not (Test-Path ".\requirements.txt")) {
    Write-Error "No se encontró requirements.txt en el repositorio."
    exit 1
}
& $venvPython -m pip install -r .\requirements.txt

Write-Host "";
Write-Host "Entorno creado e instalado correctamente." -ForegroundColor Green
Write-Host "Activa el entorno con:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "Luego puedes ejecutar tus notebooks con:" -ForegroundColor Cyan
Write-Host "  python -m notebook"
