# MoneyPrinterTurbo - daily start. Double-click start.bat.
# Starts Ollama (if needed) and the web interface, then opens the browser.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Port = 8501
$Url = "http://127.0.0.1:$Port"

function Fail($text) {
    Write-Host "[X] $text" -ForegroundColor Red
    Write-Host "See the Troubleshooting section in README-ar.md" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

function Test-Url($address) {
    try {
        Invoke-WebRequest -Uri $address -UseBasicParsing -TimeoutSec 3 | Out-Null
        return $true
    } catch { return $false }
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython) -or -not (Test-Path "config.toml")) {
    Fail "Setup is not finished. Double-click install.bat first."
}

# Temporary files and caches go to the data folder (external drive), not C:.
$DataDir = (& $VenvPython "windows\setup_config.py" --print-data-dir) 2>$null
if ($DataDir -and -not (Test-Path $DataDir)) {
    Fail "Data folder $DataDir is missing. Is the external drive connected?"
}
if ($DataDir) {
    $env:TEMP = Join-Path $DataDir "temp"
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
    $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $DataDir "playwright-browsers"
}
$models = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
if ($models) { $env:OLLAMA_MODELS = $models }

# 1) Ollama
Write-Host "Starting the AI writer (Ollama)..."
if (-not (Test-Url "http://localhost:11434/api/tags")) {
    $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $ollama) { $ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }
    if (Test-Path $ollama) {
        Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 30 -and -not (Test-Url "http://localhost:11434/api/tags"); $i++) { Start-Sleep -Seconds 1 }
    }
}
if (Test-Url "http://localhost:11434/api/tags") {
    Write-Host "  [OK] Ollama is running." -ForegroundColor Green
} else {
    Write-Host "  [!] Ollama is not running. Script writing will use the Gemini fallback if you set a key." -ForegroundColor Yellow
}

# 2) Web interface
if (Test-Url $Url) {
    Write-Host "The generator is already open."
    Start-Process $Url
    exit 0
}
Write-Host "Starting the video generator... (keep this window open while you work)"
$env:PYTHONPATH = $Root
$streamlitArgs = @(
    "-m", "streamlit", "run", "webui\Main.py",
    "--server.address=127.0.0.1", "--server.port=$Port", "--server.headless=true",
    "--browser.gatherUsageStats=False", "--client.toolbarMode=minimal",
    "--logger.hideWelcomeMessage=True", "--server.showEmailPrompt=False"
)
$server = Start-Process -FilePath $VenvPython -ArgumentList $streamlitArgs -NoNewWindow -PassThru
for ($i = 0; $i -lt 90 -and -not (Test-Url $Url); $i++) {
    if ($server.HasExited) { Fail "The web interface stopped while starting. Read the messages above." }
    Start-Sleep -Seconds 1
}
if (-not (Test-Url $Url)) { Fail "The web interface did not start within 90 seconds." }
Start-Process $Url
Write-Host ""
Write-Host "Opened $Url in your browser." -ForegroundColor Green
Write-Host "Close this window to stop the generator."
$server.WaitForExit()
