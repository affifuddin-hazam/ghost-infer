# The README's speed tables, measured again in one run on this computer:
#
#   1. a dense model that fits in VRAM (Qwen3 8B Q4_K_M) through each engine's OpenAI-compatible server, with
#      GuideLLM (scripts\bench-engines.ps1): ghost-infer, llama.cpp (defaults and set up for one user), Ollama;
#   2. a mixture-of-experts model bigger than the GPU (Qwen3-30B-A3B Q3_K_M), through each engine's server too, on
#      realistic, varied prompts (scripts\bench-moe.py: a repeated prompt would flatter ghost-infer's GPU expert
#      cache): ghost-infer; llama.cpp with the experts of some layers on the CPU (--n-cpu-moe, the way llama.cpp
#      users run these; a llama-bench sweep picks the best setting first); Ollama with its defaults.
#
# Close everything else first (above all apps that hold RAM: the MoE numbers depend on it), then:
#   powershell -ExecutionPolicy Bypass -File scripts\bench-readme.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\bench-readme.ps1 -SkipDense  (only the MoE table)
#   ... -Stage C:\ghost-bench-stage   (read the MoE model from a fast disk: it is copied there for the run, one
#        engine's copy at a time since two may not fit, and the folder is deleted at the end)
#   powershell -ExecutionPolicy Bypass -File scripts\bench-readme.ps1 -Quick      (a smoke test: minutes, numbers mean nothing)
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
    [string[]]$Engines = @('ghost-infer', 'llama.cpp', 'ollama'),  # for the MoE table
    [int]$Port = 8090,
    [string]$Stage = '',      # a new folder on a fast disk for the MoE model (see above); deleted at the end
    [switch]$SkipDense,
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
if ($Quick) { $CpuMoe = @(24) }
$out = Join-Path $root ('bench-readme-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $out, $OllamaModels | Out-Null
if ($Stage) {
    # it is deleted at the end, so it must be this script's own: a new or empty folder
    if ((Test-Path $Stage) -and (Get-ChildItem $Stage -Force | Select-Object -First 1)) { throw "$Stage exists and is not empty: give a new folder (this script deletes it at the end)" }
    $need = (Get-Item $Moe).Length + 4GB
    $free = (Get-PSDrive $Stage.Substring(0, 1)).Free
    if ($free -lt $need) { throw "$([math]::Round($free / 1GB, 1)) GB free where -Stage is; the model needs $([math]::Round($need / 1GB, 1)) GB there (with 4 GB to spare)" }
    New-Item -ItemType Directory -Force $Stage | Out-Null
}
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

if ($SkipDense) {
    Set-Content -Encoding utf8 (Join-Path $out 'dense.md') '(dense model skipped: -SkipDense)'
} else {
    Say 'dense model: GuideLLM against each server (this is the long part)'
    $denseOut = Join-Path $out 'dense'
    $args1 = @{ Model = $Dense; Tokenizer = $Tokenizer; GuideLLMPython = $GuideLLMPython; LlamaServer = $llamaServer; Ollama = $Ollama
                OllamaModels = $OllamaModels; Engines = @('ghost-infer', 'llama.cpp-1user', 'llama.cpp', 'ollama'); Port = $Port; Out = $denseOut }
    if ($Quick) { $args1.Rounds = 1; $args1.Quick = $true }
    & (Join-Path $PSScriptRoot 'bench-engines.ps1') @args1 | Tee-Object -FilePath (Join-Path $out 'dense.md')
}

# ---- 2. the mixture-of-experts model, through the servers, on realistic prompts (scripts\bench-moe.py)

$py = $GuideLLMPython  # any Python 3 runs the client; this one is at hand
$client = Join-Path $PSScriptRoot 'bench-moe.py'
$runArgs = if ($Quick) { @('--rounds', '1', '--warmup', '0', '--prompts', '2') } else { @() }
$moeOut = Join-Path $out 'moe'
New-Item -ItemType Directory -Force $moeOut | Out-Null
$labels = @()
$notes = @()
$moeRun = $Moe          # the file the engines read (a copy on -Stage while ghost-infer and llama.cpp run)
$ollamaStore = $OllamaModels
try {
if ($Stage -and ($Engines -contains 'ghost-infer' -or $Engines -contains 'llama.cpp')) {
    Say "copying the model to $Stage (for ghost-infer and llama.cpp)"
    $t = Get-Date
    Copy-Item -LiteralPath $Moe -Destination $Stage
    $moeRun = Join-Path $Stage (Split-Path $Moe -Leaf)
    Say "   copied in $([int]((Get-Date) - $t).TotalSeconds) s"
}

function Wait-GpuFree([int]$belowMiB = 1500) {
    for ($i = 0; $i -lt 60; $i++) {
        $used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($used -lt $belowMiB) { Start-Sleep 3; return }
        Start-Sleep -Milliseconds 500
    }
}
function Wait-Up([string]$url, $proc, [string]$what) {
    for ($i = 0; $i -lt 600; $i++) {
        if ($proc.HasExited) { throw "$what stopped while starting; see its log in $moeOut" }
        try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "$what did not come up in 5 minutes"
}
# stops a process this script started, with its children (Ollama runs the model in one): only that tree
function Stop-Tree($proc) { if ($proc -and -not $proc.HasExited) { & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null; $proc.WaitForExit(10000) | Out-Null } }

# starts a server, waits until $ready answers, sends it the prompts, and stops it
function Measure-Server([string]$label, [string]$exe, [string[]]$argv, [string]$ready, [string]$model) {
    Say "MoE: $label"
    $p = Start-Process -FilePath $exe -ArgumentList $argv -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $moeOut "$label.server.log") -RedirectStandardError (Join-Path $moeOut "$label.server.err.log")
    try {
        Wait-Up $ready $p $label
        & $py -I $client run --url "http://127.0.0.1:$Port" --model $model --label $label --out (Join-Path $moeOut "$label.json") @runArgs
        if ($LASTEXITCODE -ne 0) { throw "the prompts against $label failed" }
    } finally {
        Stop-Tree $p
        Wait-GpuFree
    }
}

if ($Engines -contains 'ghost-infer') {
    Measure-Server 'ghost-infer' $ghost @('serve', "`"$moeRun`"", '--gpu', '--ctx', '4096', '--port', "$Port") "http://127.0.0.1:$Port/health" 'm'
    $labels += 'ghost-infer=ghost-infer'
}

if ($Engines -contains 'llama.cpp') {
    # llama.cpp's speed does not depend on the text (each expert stays where it was put), so llama-bench's random
    # tokens are fine for picking the setting; the measured run then gets the same prompts as the others
    Say "MoE: llama.cpp, the best --n-cpu-moe of $($CpuMoe -join ', ') (llama-bench, 64 generated tokens)"
    $best = $null; $bestTg = 0
    foreach ($n in $CpuMoe) {
        $json = Join-Path $moeOut "llama-bench-ncmoe$n.json"
        $ErrorActionPreference = 'Continue'
        & $llamaBench -m $moeRun -ngl 99 -ncmoe $n -fa 1 -p 0 -n 64 -r 2 -o json 2>$null | Out-File -Encoding utf8 $json
        $ErrorActionPreference = 'Stop'
        $tg = $null
        try { $tg = ((Get-Content $json -Raw | ConvertFrom-Json) | Where-Object { $_.n_gen -gt 0 } | Select-Object -First 1).avg_ts } catch { }
        if (-not $tg) { Say "   --n-cpu-moe ${n}: did not run (out of VRAM?)"; continue }
        Say "   --n-cpu-moe ${n}: $(F1 $tg) tok/s"
        if ($tg -gt $bestTg) { $best = $n; $bestTg = $tg }
    }
    if ($null -eq $best) { throw 'no --n-cpu-moe setting ran' }
    Wait-GpuFree
    # set up for one user, its best for one person (one slot, no RAM copy of every prompt's cache)
    Measure-Server 'llama.cpp' $llamaServer @('-m', "`"$moeRun`"", '-ngl', '99', '--n-cpu-moe', "$best", '-c', '4096', '-np', '1', '--cache-ram', '0', '--port', "$Port") `
        "http://127.0.0.1:$Port/health" 'm'
    $labels += "llama.cpp=llama.cpp b11747 (--n-cpu-moe $best)"
}

if ($Engines -contains 'ollama') {
    if (Get-Process ollama*, 'ollama app' -ErrorAction SilentlyContinue) { throw 'Ollama is already running: quit it (its tray icon, Quit Ollama), then run this again.' }
    if ($Stage) {
        # its own copy goes on the stage instead: the other copy leaves first, since both may not fit
        if ($moeRun -ne $Moe) { Remove-Item -LiteralPath $moeRun; $moeRun = $Moe }
        $ollamaStore = Join-Path $Stage 'ollama'
        New-Item -ItemType Directory -Force $ollamaStore | Out-Null
    }
    $env:OLLAMA_MODELS = $ollamaStore; $env:OLLAMA_HOST = "127.0.0.1:$Port"
    # the model goes into Ollama's own store once (a copy of the file; quick when it is there already)
    $c = Start-Process -FilePath $Ollama -ArgumentList 'serve' -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $moeOut 'ollama-create.server.log')
    try {
        Wait-Up "http://127.0.0.1:$Port/api/tags" $c 'ollama'
        $mf = Join-Path $moeOut 'moe.Modelfile'
        Set-Content -Encoding ascii $mf "FROM $Moe"
        $ErrorActionPreference = 'Continue'
        & $Ollama create bench-moe -f $mf *> (Join-Path $moeOut 'ollama-create.log')
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -ne 0) { throw "ollama create failed; see $moeOut\ollama-create.log" }
    } finally { Stop-Tree $c; Wait-GpuFree }
    Measure-Server 'ollama' $Ollama @('serve') "http://127.0.0.1:$Port/api/tags" 'bench-moe'
    $labels += 'ollama=Ollama 0.40.13 (defaults)'
    $split = Select-String -Path (Join-Path $moeOut 'ollama.server.err.log') -Pattern 'CUDA0 model buffer size|CUDA_Host model buffer size|layers \(\d+ overflowing\)' | ForEach-Object { $_.Line -replace '^.*?(CUDA|\d+ layers)', '$1' }
    if ($split) { $notes += "Ollama placed the model: $(($split | Select-Object -Unique) -join '; ')." }
}

} finally {
    if ($Stage -and (Test-Path $Stage)) { Say "removing $Stage"; Remove-Item -LiteralPath $Stage -Recurse -Force }
}
if ($Stage) { $notes += "The model was read from a copy in $Stage (Ollama: its store there too), one engine at a time." }
else { $notes += "The model was read from $Moe." }

Say 'MoE: the table'
& $py -I $client summary $moeOut @labels | Out-Null
$table = Get-Content (Join-Path $moeOut 'summary.md') -Encoding UTF8
$md = @("### Mixture of experts: $(Split-Path $Moe -Leaf), realistic prompts", '',
        'Seven varied prompts (writing code, an explanation, a ~2,000-token code review, a ~1,500-token document to summarise, a chat, a word problem, SQL), each through the engine''s /v1/chat/completions, streamed, greedy, 256 tokens; one warm-up round, then two measured rounds.', '')
$md += $table
$md += '', "Free RAM at the start: $freeGb GB. GPU: $gpu.$(if ($busy.Count) { " Also running: $($busy -join ', ')." })"
$md += $notes

$md | Out-File -Encoding utf8 (Join-Path $out 'moe.md')

$all = @('# bench-readme results', '', (Get-Content (Join-Path $out 'dense.md') -Raw), '', ($md -join "`n"))
$all | Out-File -Encoding utf8 (Join-Path $out 'results.md')
Say "done: $out\results.md"
Get-Content (Join-Path $out 'results.md') | Write-Host
