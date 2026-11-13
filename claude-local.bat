@echo off
title Claude Code (local)
rem Claude Code on the local engine (start-engine.bat first). Your normal `claude` is untouched.
rem it works in its own folder, so what it writes stays out of other projects
if not exist "E:\AI\claude-local-demo" mkdir "E:\AI\claude-local-demo"
cd /d "E:\AI\claude-local-demo"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0claude-local.ps1" %*
