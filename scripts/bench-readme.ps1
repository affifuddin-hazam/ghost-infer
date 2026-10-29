# The README's speed tables, measured again in one run on this computer:
#
#   1. a dense model that fits in VRAM (Qwen3 8B Q4_K_M) through each engine's OpenAI-compatible server, with
#      GuideLLM (scripts\bench-engines.ps1): ghost-infer, llama.cpp (defaults and set up for one user), Ollama;
#   2. a mixture-of-experts model bigger than the GPU (Qwen3-30B-A3B Q3_K_M), with each engine's own timing:
#      ghost-infer `bench`; llama.cpp's llama-bench with the experts of some layers on the CPU (--n-cpu-moe, the
#      way llama.cpp users run these; the best setting of a sweep counts); Ollama with its defaults, through its API.
#
# Close everything else first (above all apps that hold RAM: the MoE numbers depend on it), then:
#   powershell -ExecutionPolicy Bypass -File scripts\bench-readme.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\bench-readme.ps1 -Quick     (a smoke test: minutes, numbers mean nothing)
#
# Everything goes to bench-readme-<time>\ (gitignored): the raw results, and results.md with both tables.
param(
    [string]$Dense = '',      # default: models\JOSIEFIED-Qwen3-8B-Q4_K_M.gguf
    [string]$Tokenizer = 'Qwen/Qwen3-8B',
    [string]$Moe = '',        # default: models\Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf
    [string]$GuideLLMPython = 'E:\AI\guidellm-venv\Scripts\python.exe',
    [string]$LlamaDir = 'E:\AI\llama.cpp-b11747',
    [string]$Ollama = 'E:\AI\ollama-0.40.13\ollama.exe',
    [string]$OllamaModels = 'E:\AI\ollama-bench-models',  # its private store (a model is copied in: keep it off a full C:)
    [int[]]$CpuMoe = @(16, 20, 24, 28, 32, 36, 40, 48),   # llama.cpp: layers whose experts stay on the CPU, tried in turn
    [int]$Port = 8090,
    [switch]$Quick
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $Dense) { $Dense = Join-Path $root 'models\JOSIEFIED-Qwen3-8B-Q4_K_M.gguf' }
if (-not $Moe) { $Moe = Join-Path $root 'models\Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf' }
$ghost = Join-Path $root 'target\release\ghost-infer.exe'
$llamaServer = Join-Path $LlamaDir 'llama-server.exe'
$llamaBench = Join-Path $LlamaDir 'llama-bench.exe'
foreach ($f in @($Dense, $Moe, $ghost, $GuideLLMPython, $llamaServer, $llamaBench, $Ollama)) {
    if (-not (Test-Path $f)) { throw "not found: $f" }
}
if ($Quick) { $CpuMoe = @(28, 48) }
$out = Join-Path $root ('bench-readme-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $out, $OllamaModels | Out-Null
function Say($text) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $text" }
$inv = [Globalization.CultureInfo]::InvariantCulture
function F1($x) { [string]::Format($inv, '{0:0.0}', [double]$x) }

# what the computer was doing: the MoE numbers depend on free RAM
$os = Get-CimInstance Win32_OperatingSystem
$freeGb = F1 ($os.FreePhysicalMemory / 1MB)
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader) -join ' '
$busy = @(Get-Process claude*, chrome, msedge, firefox -ErrorAction SilentlyContinue | Group-Object ProcessName | ForEach-Object { $_.Name })
Say "free RAM $freeGb GB; GPU $gpu"
if ($busy.Count) { Say "note: also running: $($busy -join ', ') (they hold RAM; close them for the real numbers)" }

# ---- 1. the dense model, through the servers (GuideLLM)

Say 'dense model: GuideLLM against each server (this is the long part)'
$denseOut = Join-Path $out 'dense'
$args1 = @{ Model = $Dense; Tokenizer = $Tokenizer; GuideLLMPython = $GuideLLMPython; LlamaServer = $llamaServer; Ollama = $Ollama
            OllamaModels = $OllamaModels; Engines = @('ghost-infer', 'llama.cpp-1user', 'llama.cpp', 'ollama'); Port = $Port; Out = $denseOut }
if ($Quick) { $args1.Rounds = 1; $args1.Quick = $true }
& (Join-Path $PSScriptRoot 'bench-engines.ps1') @args1 | Tee-Object -FilePath (Join-Path $out 'dense.md')

# ---- 2. the mixture-of-experts model, each engine's own timing

$moeRows = @()
Say 'MoE: ghost-infer bench'
foreach ($n in @(256, 1024)) {
    $line = & $ghost bench $Moe --prompt-len $n -n 128 --gpu 2>$null | Select-String 'generate'
    "$line" | Out-File -Append -Encoding utf8 (Join-Path $out 'moe-ghost.txt')
    if ("$line" -match 'prompt \d+ tok:\s*([\d.]+) tok/s \| generate \d+ tok:\s*([\d.]+)') {
        $moeRows += [pscustomobject]@{ engine = 'ghost-infer'; setup = 'default'; prompt = $n; bucket = $n; pp = [double]$Matches[1]; tg = [double]$Matches[2] }
    }
    if ($Quick) { break }
}

Say "MoE: llama.cpp llama-bench, experts of N layers on the CPU, N = $($CpuMoe -join ', ')"
foreach ($n in $CpuMoe) {
    $json = Join-Path $out "moe-llama-ncmoe$n.json"
    $ErrorActionPreference = 'Continue'
    & $llamaBench -m $Moe -ngl 99 -ncmoe $n -fa 1 -p 256,1024 -n 128 -r 3 -o json 2>$null | Out-File -Encoding utf8 $json
    $ErrorActionPreference = 'Stop'
    try { $runs = Get-Content $json -Raw | ConvertFrom-Json } catch { Say "   --n-cpu-moe $n did not run (out of VRAM?)"; continue }
    foreach ($r in $runs) {
        $moeRows += [pscustomobject]@{ engine = 'llama.cpp'; setup = "--n-cpu-moe $n"; prompt = [int]$r.n_prompt; bucket = [int]$r.n_prompt; pp = $(if ($r.n_prompt -gt 0) { [double]$r.avg_ts } else { 0 }); tg = $(if ($r.n_gen -gt 0) { [double]$r.avg_ts } else { 0 }) }
    }
    $tg = ($runs | Where-Object { $_.n_gen -gt 0 } | Select-Object -First 1).avg_ts
    Say "   --n-cpu-moe ${n}: generation $(F1 $tg) tok/s"
}

Say 'MoE: Ollama with its defaults'
$env:OLLAMA_MODELS = $OllamaModels; $env:OLLAMA_HOST = "127.0.0.1:$Port"
if (Get-Process ollama*, 'ollama app' -ErrorAction SilentlyContinue) { throw 'Ollama is already running: quit it (its tray icon, Quit Ollama), then run this again.' }
$ol = Start-Process -FilePath $Ollama -ArgumentList 'serve' -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $out 'moe-ollama.server.log')
try {
    for ($i = 0; $i -lt 120; $i++) { try { Invoke-RestMethod "http://127.0.0.1:$Port/api/tags" | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }
    $mf = Join-Path $out 'moe.Modelfile'
    Set-Content -Encoding ascii $mf "FROM $Moe"
    $ErrorActionPreference = 'Continue'
    & $Ollama create bench-moe -f $mf *> (Join-Path $out 'moe-ollama-create.log')
    $ErrorActionPreference = 'Stop'
    $sentence = 'The quick brown fox jumps over the lazy dog while the engine reads the prompt. '  # about 16 tokens
    foreach ($n in @(256, 1024)) {
        for ($k = 1; $k -le $(if ($Quick) { 1 } else { 3 }); $k++) {
            # a different first line each time, so Ollama's prompt cache cannot reuse the last one
            $prompt = "Run $k of $n, $([guid]::NewGuid()). " + ($sentence * [int]($n / 16))
            $body = @{ model = 'bench-moe'; prompt = $prompt; raw = $true; stream = $false; options = @{ num_predict = 128; num_ctx = 4096; temperature = 0 } } | ConvertTo-Json -Depth 4
            $r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/generate" -ContentType 'application/json' -Body $body -TimeoutSec 900
            $moeRows += [pscustomobject]@{ engine = 'Ollama'; setup = 'defaults'; prompt = [int]$r.prompt_eval_count; bucket = $n
                pp = [double]$r.prompt_eval_count / ([double]$r.prompt_eval_duration / 1e9); tg = [double]$r.eval_count / ([double]$r.eval_duration / 1e9) }
            Say "   $($r.prompt_eval_count)-token prompt: generation $(F1 ([double]$r.eval_count / ([double]$r.eval_duration / 1e9))) tok/s"
        }
        if ($Quick) { break }
    }
    $ps = (Invoke-RestMethod "http://127.0.0.1:$Port/api/ps").models | Select-Object -First 1
    if ($ps) { Say "   Ollama put $([math]::Round($ps.size_vram / 1GB, 1)) of $([math]::Round($ps.size / 1GB, 1)) GB on the GPU" }
} finally {
    # the server this script started and its model runners: only that tree
    if (-not $ol.HasExited) { & taskkill.exe /PID $ol.Id /T /F 2>&1 | Out-Null }
}
$moeRows | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $out 'moe.csv')

# ---- the MoE table: generation as the range over runs and prompt lengths, prompt reading per length (llama.cpp: its best --n-cpu-moe)

$bestLlama = $moeRows | Where-Object { $_.engine -eq 'llama.cpp' -and $_.tg -gt 0 } | Sort-Object tg -Descending | Select-Object -First 1
$cols = @(
    @{ name = 'ghost-infer'; rows = @($moeRows | Where-Object { $_.engine -eq 'ghost-infer' }) },
    @{ name = $(if ($bestLlama) { "llama.cpp b11747 ($($bestLlama.setup))" } else { 'llama.cpp b11747' }); rows = @($moeRows | Where-Object { $_.engine -eq 'llama.cpp' -and $bestLlama -and $_.setup -eq $bestLlama.setup }) },
    @{ name = 'Ollama 0.40.13'; rows = @($moeRows | Where-Object { $_.engine -eq 'Ollama' }) }
)
function Range($xs) { $xs = @($xs | Where-Object { $_ -gt 0 }); if (-not $xs.Count) { return '-' }; $lo = [math]::Round(($xs | Measure-Object -Minimum).Minimum); $hi = [math]::Round(($xs | Measure-Object -Maximum).Maximum); if ($lo -eq $hi) { "$lo" } else { "$lo-$hi" } }
$md = @("### Mixture of experts: $(Split-Path $Moe -Leaf), tokens per second", '', "| | $(($cols | ForEach-Object { $_.name }) -join ' | ') |", "|---|$('---|' * $cols.Count)")
$md += "| Generation | $(($cols | ForEach-Object { Range ($_.rows | ForEach-Object { $_.tg }) }) -join ' | ') |"
foreach ($n in @(256, 1024)) {
    $md += "| Prompt, $n tokens | $(($cols | ForEach-Object { Range ($_.rows | Where-Object { $_.bucket -eq $n } | ForEach-Object { $_.pp }) }) -join ' | ') |"
}
$md += '', "Free RAM at the start: $freeGb GB. GPU: $gpu.$(if ($busy.Count) { " Also running: $($busy -join ', ')." })"
$md | Out-File -Encoding utf8 (Join-Path $out 'moe.md')

$all = @('# bench-readme results', '', (Get-Content (Join-Path $out 'dense.md') -Raw), '', ($md -join "`n"))
$all | Out-File -Encoding utf8 (Join-Path $out 'results.md')
Say "done: $out\results.md"
Get-Content (Join-Path $out 'results.md') | Write-Host
