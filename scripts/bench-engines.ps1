# Engine benchmark with GuideLLM (https://github.com/vllm-project/guidellm): ghost-infer vs llama.cpp's
# llama-server vs Ollama on the same GGUF file, one user at a time (synchronous). Each server runs alone on
# the GPU, every workload runs against each server in turn, and -Rounds 2 repeats the whole set so GPU clock
# drift between engines averages out. Results: one GuideLLM JSON per run, and Markdown tables at the end
# (scripts\bench-compare.py).
#
#   .\scripts\bench-engines.ps1 -Model models\Qwen3-8B-Q4_K_M.gguf -Tokenizer Qwen/Qwen3-8B `
#       -LlamaServer C:\llama.cpp\llama-server.exe -GuideLLMPython C:\guidellm-venv\Scripts\python.exe
#   .\scripts\bench-engines.ps1 ... -Engines ghost-infer,llama.cpp -Rounds 1
param(
    [Parameter(Mandatory)] [string]$Model,
    [Parameter(Mandatory)] [string]$Tokenizer,          # Hugging Face id of the model's tokenizer, for GuideLLM's synthetic prompts
    [Parameter(Mandatory)] [string]$GuideLLMPython,     # python.exe of an environment with guidellm installed
    [string]$LlamaServer = '',
    [string]$Ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
    [string]$OllamaModels = (Join-Path $env:TEMP 'ghost-infer-bench-ollama'),  # a private store for the bench copies
    [string[]]$Engines = @('ghost-infer', 'llama.cpp', 'ollama'),
    [int]$Rounds = 2,
    [int]$Port = 8090,
    [string]$Out = '',
    [switch]$Quick,  # a smoke test: 3 requests per workload (the numbers mean nothing)
    [int]$Round = 0  # internal: run one engine for this round, in a process of its own
)
$ErrorActionPreference = 'Stop'

# name, prompt tokens, output tokens, requests
$workloads = @(
    @{ name = 'chat'; prompt = 256;  output = 128; requests = 24 },
    @{ name = 'long'; prompt = 2048; output = 256; requests = 12 }
)
if ($Quick) { foreach ($w in $workloads) { $w.requests = 3 } }
$root = Split-Path $PSScriptRoot -Parent
$ghost = Join-Path $root 'target\release\ghost-infer.exe'
$Model = (Resolve-Path $Model).Path
if (-not $Out) { $Out = Join-Path $root ('bench-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $Out | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONWARNINGS = 'ignore'
$stem = [IO.Path]::GetFileNameWithoutExtension($Model)
if ($Engines -contains 'ghost-infer' -and -not (Test-Path $ghost)) { throw "build first: cargo build --release" }
if (($Engines -like 'llama.cpp*') -and (-not $LlamaServer -or -not (Test-Path $LlamaServer))) { throw "pass -LlamaServer <path to llama-server.exe>" }

function Wait-Url([string]$url, $proc, [int]$secs = 300) {
    $deadline = (Get-Date).AddSeconds($secs)  # a cold load from a slow disk can take a minute
    while ((Get-Date) -lt $deadline) {
        if ($proc -and $proc.HasExited) { return $false }
        try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true } catch { Start-Sleep -Milliseconds 500 }
    }
    return $false
}
# the next server must not start before the driver has freed the last one's memory, or it sizes itself to what is left
function Wait-GpuFree([int]$belowMiB = 1500) {
    for ($i = 0; $i -lt 60; $i++) {
        $used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($used -lt $belowMiB) { Start-Sleep 5; return }  # and a moment more for the driver to finish tearing the context down
        Start-Sleep -Milliseconds 500
    }
}
# Stops a server this script started, with the processes it started in turn (Ollama runs its models in child
# processes); only that tree, never another program of the same name.
function Stop-Tree($proc) {
    if ($proc -and -not $proc.HasExited) { & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null; $proc.WaitForExit(10000) | Out-Null }
    Start-Sleep 1
}
# an Ollama the user runs would compete for the GPU: it is theirs to close, not this script's
if ($Round -eq 0 -and $Engines -contains 'ollama' -and (Get-Process ollama*, 'ollama app' -ErrorAction SilentlyContinue)) {
    throw 'Ollama is already running: quit it (its tray icon, Quit Ollama), then run this again.'
}

# Ollama ignores max_completion_tokens (what GuideLLM sends), so each output length gets a model with num_predict set.
# The Modelfile also gives a ChatML template, for GGUF files whose template Ollama does not recognise.
if ($Round -eq 0 -and $Engines -contains 'ollama') {
    $env:OLLAMA_MODELS = $OllamaModels
    $env:OLLAMA_HOST = "127.0.0.1:$Port"
    $p = Start-Process -FilePath $Ollama -ArgumentList 'serve' -PassThru -WindowStyle Hidden -RedirectStandardError "$Out\ollama-create.err.log"
    if (-not (Wait-Url "http://127.0.0.1:$Port/api/tags" $p)) { throw 'ollama serve did not start' }
    foreach ($n in ($workloads | ForEach-Object { $_.output } | Sort-Object -Unique)) {
        $mf = Join-Path $Out "ollama-$n.Modelfile"
        @"
FROM $Model
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>
PARAMETER num_predict $n
"@ | Set-Content -Encoding ascii $mf
        $ErrorActionPreference = 'Continue'  # its progress goes to stderr
        & $Ollama create "bench-$($stem.ToLower())-$n" -f $mf *> "$Out\ollama-create-$n.log"
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -ne 0) { Stop-Tree $p; throw "ollama create failed for $n; see $Out\ollama-create-$n.log" }
    }
    Stop-Tree $p
}

function Start-Engine([string]$engine, [string]$tag) {
    $log = @{ RedirectStandardOutput = "$Out\$tag.server.log"; RedirectStandardError = "$Out\$tag.server.err.log" }
    switch ($engine) {
        'ghost-infer' {
            $proc = Start-Process -FilePath $ghost -ArgumentList @('serve', $Model, '--gpu', '--port', $Port) -PassThru -WindowStyle Hidden @log
            $url = "http://127.0.0.1:$Port/health"
        }
        'llama.cpp' {
            # every layer on the GPU, an 8k context (both workloads fit), ChatML; everything else llama.cpp's defaults
            $proc = Start-Process -FilePath $LlamaServer -ArgumentList @('-m', $Model, '-ngl', '99', '-c', '8192', '--chat-template', 'chatml', '--port', $Port) `
                -PassThru -WindowStyle Hidden @log
            $url = "http://127.0.0.1:$Port/health"
        }
        'llama.cpp-1user' {
            # the same, set up for one user: one slot, and no RAM copy of every prompt's cache (by default llama-server keeps
            # up to 8 GB of them, which costs time on each request and buys nothing when prompts do not repeat)
            $proc = Start-Process -FilePath $LlamaServer -ArgumentList @('-m', $Model, '-ngl', '99', '-c', '8192', '--chat-template', 'chatml', '--port', $Port, '-np', '1', '--cache-ram', '0') `
                -PassThru -WindowStyle Hidden @log
            $url = "http://127.0.0.1:$Port/health"
        }
        'ollama' {
            $env:OLLAMA_MODELS = $OllamaModels
            $env:OLLAMA_HOST = "127.0.0.1:$Port"
            $proc = Start-Process -FilePath $Ollama -ArgumentList 'serve' -PassThru -WindowStyle Hidden @log
            $url = "http://127.0.0.1:$Port/api/tags"
        }
    }
    if (-not (Wait-Url $url $proc)) { throw "$tag did not come up; see $Out\$tag.server.err.log" }
    return $proc
}

# Each engine runs in a PowerShell process of its own. In one long-lived process, llama-server started right after
# another engine kept dying during CUDA start-up (never when started from a fresh process); a fresh process per engine avoids it.
if ($Round -eq 0) {
    for ($r = 1; $r -le $Rounds; $r++) {
        foreach ($engine in $Engines) {
            $child = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath, '-Model', $Model, '-Tokenizer', $Tokenizer,
                '-GuideLLMPython', $GuideLLMPython, '-Ollama', $Ollama, '-OllamaModels', $OllamaModels, '-Engines', $engine, '-Port', $Port, '-Out', $Out, '-Round', $r)
            if ($LlamaServer) { $child += '-LlamaServer', $LlamaServer }
            if ($Quick) { $child += '-Quick' }
            & powershell @child
            if ($LASTEXITCODE -ne 0) { throw "$engine round $r failed" }
        }
    }
    & $GuideLLMPython -I (Join-Path $PSScriptRoot 'bench-compare.py') $Out
    return
}

foreach ($r in @($Round)) {
    foreach ($engine in $Engines) {
        $tag = "$engine-r$r"
        Write-Host "== $tag" -ForegroundColor Cyan
        # a server that dies while starting gets two more tries
        for ($try = 1; ; $try++) {
            try { $proc = Start-Engine $engine $tag; break }
            catch { if ($try -ge 3) { throw }; Write-Host "   $($_.Exception.Message); trying again"; Wait-GpuFree; Start-Sleep 5 }
        }
        try {
            foreach ($w in $workloads) {
                $model = if ($engine -eq 'ollama') { "bench-$($stem.ToLower())-$($w.output)" } else { $stem }
                $validate = if ($engine -eq 'ollama') { ',validate_backend=false' } else { '' }  # Ollama has no /health
                $run = "$engine-$($w.name)-r$r"
                Write-Host "   $($w.name): $($w.prompt) in / $($w.output) out, $($w.requests) requests"
                $ErrorActionPreference = 'Continue'  # PowerShell 5.1 turns native stderr into terminating errors under 'Stop'
                & $GuideLLMPython -m guidellm run `
                    --backend "kind=openai_http,target=http://127.0.0.1:$Port,model=$model,request_format=/v1/chat/completions$validate" `
                    --profile 'kind=synchronous,warmup=0.1' `
                    --constraint "kind=max_requests,count=$($w.requests)" `
                    --data "kind=synthetic_text,prompt_tokens=$($w.prompt),output_tokens=$($w.output)" `
                    --tokenizer "kind=huggingface_auto,model=$Tokenizer" `
                    --output "kind=json,path=$Out\$run.json" `
                    --disable-console *> "$Out\$run.guidellm.log"
                $ErrorActionPreference = 'Stop'
                if ($LASTEXITCODE -ne 0) { throw "$run GuideLLM failed; see $Out\$run.guidellm.log" }
            }
        } finally {
            Stop-Tree $proc
            Wait-GpuFree
        }
    }
}
