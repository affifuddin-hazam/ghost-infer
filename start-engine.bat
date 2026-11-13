@echo off
title ghost-infer engine
rem Starts the ghost-infer engine with the Claude-distilled Qwen3.5 35B-A3B (for claude-local.bat). Keep this
rem window open; close it (or Ctrl+C) to stop the engine and free the GPU.
set EXE=E:\AI\betterllama\target\release\ghost-infer.exe
set MODEL=C:\models\Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated.i1-Q3_K_M.gguf
"%EXE%" serve "%MODEL%" --gpu --ctx 65536 --kv q4 --models C:\models --models D:\models --models E:\AI\betterllama\models
pause
