@echo off
title ghost-infer monitor
rem Live tok/s graph of the running engine (start-engine.bat). Ctrl+C or close the window to quit.
rem (target\mon: a build beside the engine's, made while the engine was running from target\release)
set EXE=E:\AI\betterllama\target\release\ghost-infer.exe
if exist "E:\AI\betterllama\target\mon\release\ghost-infer.exe" set EXE=E:\AI\betterllama\target\mon\release\ghost-infer.exe
"%EXE%" monitor 127.0.0.1:8080
