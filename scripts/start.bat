@echo off
cd /d "%~dp0\.."
D:\Anaconda\python.exe -m pip show neo4j >nul 2>nul
if errorlevel 1 D:\Anaconda\python.exe -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -Command "$svc=Get-Service neo4j -ErrorAction SilentlyContinue; if ($svc -and $svc.Status -ne 'Running') { Start-Service neo4j }; $deadline=(Get-Date).AddSeconds(30); while ((Get-Date) -lt $deadline) { if (Test-NetConnection 127.0.0.1 -Port 7687 -InformationLevel Quiet) { exit 0 }; Start-Sleep -Seconds 1 }; exit 0"
D:\Anaconda\python.exe run.py
