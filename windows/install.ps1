# MoneyPrinterTurbo - first-time setup for Windows 10/11.
# Run it by double-clicking install.bat in the project folder.
# Messages are in simple English on purpose: the Windows console cannot show
# Arabic correctly. README-ar.md explains every step in Arabic.

# "Continue": in Windows PowerShell 5.1, "Stop" turns harmless stderr output
# of native programs (pip, winget, ollama) into fatal errors.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PythonVersion = "3.11"
$DefaultModel = "gemma3:4b"
$BetterModel = "gemma4:e4b"

function Step($number, $text) {
    Write-Host ""
    Write-Host "==== STEP $number : $text ====" -ForegroundColor Cyan
}
function Ok($text) { Write-Host "  [OK] $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [!] $text" -ForegroundColor Yellow }
function Fail($text) {
    Write-Host ""
    Write-Host "  [X] $text" -ForegroundColor Red
    Write-Host "  See the Troubleshooting section in README-ar.md" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

function Refresh-Path {
    # winget installs update PATH in the registry, not in this window.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Has-Winget { return [bool](Get-Command winget -ErrorAction SilentlyContinue) }

function Winget-Install($id, $name) {
    if (-not (Has-Winget)) {
        Warn "winget is not available, so $name cannot be installed automatically."
        return $false
    }
    Write-Host "  Installing $name with winget (this can take a few minutes)..."
    winget install -e --id $id --accept-package-agreements --accept-source-agreements --silent
    Refresh-Path
    return $true
}

function Find-Python {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $version = & py "-$PythonVersion" -c "import sys; print(sys.version.split()[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $version) { return ,@("py", "-$PythonVersion") }
    }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return ,@($candidate) }
    }
    return $null
}

function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $default = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $default) { return $default }
    return $null
}

function Test-Ollama {
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
        return $true
    } catch { return $false }
}

Write-Host "=============================================================="
Write-Host "   MoneyPrinterTurbo - free long video generator - SETUP"
Write-Host "=============================================================="
Write-Host "Project folder: $Root"
if ($Root.Substring(0, 1).ToUpper() -eq "C") {
    Warn "The project is on drive C:. Your C: drive is almost full."
    Warn "It is better to put the project folder on your external drive (for example E:\MoneyPrinterTurbo)."
    $answer = Read-Host "  Continue anyway? (y/N)"
    if ($answer -notmatch "^[yY]") { exit 0 }
}

# ---------------------------------------------------------------------------
Step 1 "Data folder (videos, cache, music, AI model)"
$suggested = Join-Path $Root "storage"
foreach ($drive in @("E", "D", "F")) {
    if (Test-Path "${drive}:\") {
        $suggested = "${drive}:\MoneyPrinterData"
        break
    }
}
$existing = ""
if ((Test-Path "config.toml") -and (Test-Path ".venv\Scripts\python.exe")) {
    $existing = (& ".venv\Scripts\python.exe" "windows\setup_config.py" --print-data-dir) 2>$null
    if ($existing) { $suggested = $existing }
}
Write-Host "  Everything heavy (videos, downloaded clips, the AI model ~3-7 GB) goes here."
$DataDir = Read-Host "  Data folder [press Enter for $suggested]"
if ([string]::IsNullOrWhiteSpace($DataDir)) { $DataDir = $suggested }
$DataDir = $DataDir.Trim().Trim('"')
try { New-Item -ItemType Directory -Force -Path $DataDir -ErrorAction Stop | Out-Null } catch { Fail "Cannot create folder $DataDir" }
foreach ($sub in @("temp", "pip-cache", "ollama-models", "music", "branding", "playwright-browsers")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DataDir $sub) | Out-Null
}
# Keep every temporary and cache file off the small C: drive.
$env:TEMP = Join-Path $DataDir "temp"
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = "1"  # Python reads/writes text as UTF-8 (Arabic), not the Windows code page
$env:PIP_CACHE_DIR = Join-Path $DataDir "pip-cache"
# The browser that reads websites (AI Website Video) lives on the data drive too.
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $DataDir "playwright-browsers"
$ModelsDir = Join-Path $DataDir "ollama-models"
Ok "Data folder: $DataDir"

# ---------------------------------------------------------------------------
Step 2 "Python $PythonVersion"
$Python = Find-Python
if (-not $Python) {
    Warn "Python $PythonVersion was not found."
    if (Winget-Install "Python.Python.3.11" "Python 3.11") { $Python = Find-Python }
}
if (-not $Python) {
    Start-Process "https://www.python.org/downloads/release/python-3119/"
    Fail "Install Python 3.11 (tick 'Add python.exe to PATH'), then run install.bat again."
}
$pyExe = $Python[0]
$pyArgs = @()
if ($Python.Length -gt 1) { $pyArgs = $Python[1..($Python.Length - 1)] }
Ok ("Python: " + (& $pyExe @pyArgs -c "import sys; print(sys.version.split()[0])"))

# ---------------------------------------------------------------------------
Step 3 "FFmpeg (video tool)"
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Ok "FFmpeg found."
} else {
    Winget-Install "Gyan.FFmpeg" "FFmpeg" | Out-Null
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Ok "FFmpeg installed." }
    else { Warn "FFmpeg not on PATH. The built-in copy from the Python packages will be used instead (works fine)." }
}

# ---------------------------------------------------------------------------
Step 4 "Ollama (runs the AI writer on your laptop)"
# Store AI models on the data drive, not on C:. Saved for future sessions too.
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelsDir, "User")
$env:OLLAMA_MODELS = $ModelsDir
$Ollama = Find-Ollama
if (-not $Ollama) {
    if (Winget-Install "Ollama.Ollama" "Ollama") { $Ollama = Find-Ollama }
}
if (-not $Ollama) {
    Start-Process "https://ollama.com/download/windows"
    Fail "Install Ollama from the page that just opened, then run install.bat again."
}
Ok "Ollama: $Ollama"
if (Test-Ollama) {
    # An Ollama started before OLLAMA_MODELS was set would save the model on C:.
    Warn "Ollama is already running. Restarting it so models are saved in $ModelsDir"
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}
Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden
for ($i = 0; $i -lt 30 -and -not (Test-Ollama); $i++) { Start-Sleep -Seconds 1 }
if (-not (Test-Ollama)) { Fail "Ollama did not start." }

# ---------------------------------------------------------------------------
Step 5 "Choose the AI writer model"
Write-Host "  1) $DefaultModel  - recommended: good Arabic and English, ~3.3 GB, fastest on your CPU"
Write-Host "  2) $BetterModel - better writing, bigger and about 2x slower"
$choice = Read-Host "  Your choice [1]"
$Model = $DefaultModel
if ($choice -eq "2") { $Model = $BetterModel }
Write-Host "  Downloading $Model into $ModelsDir (only the first time; this is big, be patient)..."
& $Ollama pull $Model
if ($LASTEXITCODE -ne 0) { Fail "Could not download $Model. Check your internet and run install.bat again." }
Ok "Model ready: $Model"

# ---------------------------------------------------------------------------
Step 6 "Python packages (fixed versions)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pyExe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the Python environment (.venv)." }
}
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r "windows\requirements-lock.txt"
if ($LASTEXITCODE -ne 0) { Fail "Package installation failed. Check your internet and run install.bat again." }
Ok "Packages installed."
Write-Host "  Installing the browser that reads websites (about 150 MB, into $env:PLAYWRIGHT_BROWSERS_PATH)..."
& $VenvPython -m playwright install chromium
if ($LASTEXITCODE -ne 0) { Warn "Browser install failed. AI Website Video needs it: run install.bat again later." }
else { Ok "Website browser installed." }

# ---------------------------------------------------------------------------
Step 7 "Your free API keys (stored only on this laptop)"
Write-Host "  Pexels key gives free stock videos. Get one at https://www.pexels.com/api/ (see README-ar.md)."
Write-Host "  Press Enter to keep the saved value, type - to remove it."
$Pexels = Read-Host "  Pexels API key"
$Pixabay = Read-Host "  Pixabay API key (optional)"
Write-Host "  Optional: a free Google Gemini key is used ONLY if Ollama fails (https://aistudio.google.com/apikey)."
$Gemini = Read-Host "  Gemini API key (optional)"
Write-Host "  Video quality: 1) 720p - about 2x faster (recommended for your laptop)  2) 1080p"
$quality = Read-Host "  Your choice [1]"
$Resolution = "720p"
if ($quality -eq "2") { $Resolution = "1080p" }

# Windows PowerShell drops empty arguments, so only pass keys that were typed.
$cfgArgs = @("windows\setup_config.py", "--data-dir", $DataDir, "--model", $Model, "--resolution", $Resolution)
if ($Pexels.Trim()) { $cfgArgs += @("--pexels-key", $Pexels.Trim()) }
if ($Pixabay.Trim()) { $cfgArgs += @("--pixabay-key", $Pixabay.Trim()) }
if ($Gemini.Trim()) { $cfgArgs += @("--gemini-key", $Gemini.Trim()) }
& $VenvPython @cfgArgs
if ($LASTEXITCODE -ne 0) { Fail "Could not write config.toml." }

# ---------------------------------------------------------------------------
Step 8 "Quick self-test"
& $VenvPython -c "import moviepy, streamlit, edge_tts, arabic_reshaper, bidi; print('imports ok')"
if ($LASTEXITCODE -ne 0) { Fail "Self-test failed." }
Ok "Self-test passed."

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "  DONE! Double-click start.bat to open the video generator."   -ForegroundColor Green
Write-Host "  Put your own music (mp3) in: $(Join-Path $DataDir 'music')"  -ForegroundColor Green
Write-Host "  Finished videos are saved in: $(Join-Path $DataDir 'tasks')" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Read-Host "Press Enter to close"
