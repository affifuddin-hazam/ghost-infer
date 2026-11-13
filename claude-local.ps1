# Claude Code on the local engine: ghost-infer (start-engine.bat) behind claude-code-router, which turns Claude
# Code's Anthropic requests into the engine's OpenAI ones. Its own config folder (CLAUDE_CONFIG_DIR), so the
# normal `claude` (its login, settings, history) is untouched. The status line shows the engine's tok/s.
# Arguments go to claude (e.g. claude-local.bat -p "hello").

$cfg = Join-Path $env:LOCALAPPDATA 'ghost-infer\claude-local'
New-Item -ItemType Directory -Force $cfg | Out-Null
$status = 'E:\AI\betterllama\scripts\claude-local\statusline.js'
$settings = Join-Path $cfg 'settings.json'
if (-not (Test-Path $settings)) {
    @{
        # refreshInterval (seconds, 1 the least): the speed updates while a reply is written, not only after it
        statusLine = @{ type = 'command'; command = "node `"$($status -replace '\\', '/')`""; padding = 0; refreshInterval = 1 }
    } | ConvertTo-Json -Depth 5 | Set-Content $settings -Encoding utf8
}

# the engine must be up (start-engine.bat)
try { $null = Invoke-RestMethod http://127.0.0.1:8080/api/speed -TimeoutSec 3 }
catch { Write-Host 'The engine is not running: start start-engine.bat first.' -ForegroundColor Yellow; exit 1 }

# the router (port 3456), in the background if it is not up yet
if (-not (Get-NetTCPConnection -LocalPort 3456 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath (Join-Path $env:APPDATA 'npm\ccr.cmd') -ArgumentList 'start' -WindowStyle Hidden
    for ($i = 0; $i -lt 40 -and -not (Get-NetTCPConnection -LocalPort 3456 -State Listen -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 250 }
}

# only this process and its children see these
$env:CLAUDE_CONFIG_DIR = $cfg
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:3456'
$env:ANTHROPIC_AUTH_TOKEN = 'local'
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
$env:ANTHROPIC_MODEL = 'ghost,Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated.i1-Q3_K_M'
$env:ANTHROPIC_SMALL_FAST_MODEL = $env:ANTHROPIC_MODEL
# no side requests (titles, tips): each would make the engine read a different prompt and lose its cache
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
$env:DISABLE_NON_ESSENTIAL_MODEL_CALLS = '1'
$env:DISABLE_TELEMETRY = '1'
$env:API_TIMEOUT_MS = '900000'
# the engine's real window (start-engine.bat: --ctx 65536), so Claude Code compacts in time
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = '65536'

& claude @args
