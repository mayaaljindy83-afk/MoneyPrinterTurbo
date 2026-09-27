@echo off
rem Daily start: opens the video generator in your browser.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\start.ps1"
if errorlevel 1 pause
