# The README's numbers again, for ghost-infer only (llama.cpp's and Ollama's from 2026-11-10 still stand: their
# versions have not changed), in one run on this computer:
#
#   1. a dense model that fits in VRAM (Qwen3 8B Q4_K_M): GuideLLM through the server (scripts\bench-engines.ps1),
#      as the README's first row;
#   2. mixture-of-experts models bigger than the GPU: the seven varied prompts through the server
#      (scripts\bench-moe.py via bench-readme.ps1 -Engines ghost-infer), one model after another: the README's
#      Qwen3.6-35B-A3B and Qwen3-30B-A3B rows, and the Claude-distilled Qwen3.5-35B-A3B the app uses now;
#   3. long contexts on Qwen3.6-35B-A3B: writing after a 64K-token prompt, and reading a 128K-token one
#      (ghost-infer bench --prompt-len).
#
# Close everything else first (above all the app, the engine and anything holding RAM or the GPU), then:
#   powershell -ExecutionPolicy Bypass -File scripts\bench-ghost.ps1
#   ... -Quick          a smoke test: a few minutes, the numbers mean nothing
#   ... -SkipDense / -SkipMoe / -SkipLong   leave a part out
#
# Everything goes to bench-ghost-<time>\ (gitignored with bench-*): each part's raw results and results.md.
param(
    [string]$Dense = '',
    [string[]]$Moe = @(
        'D:\models\Huihui-Qwen3.6-35B-A3B-abliterated.i1-Q3_K_M.gguf',
        'E:\AI\betterllama\models\Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf',
        'C:\models\Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated.i1-Q3_K_M.gguf'
    ),
    [string]$Long = 'D:\models\Huihui-Qwen3.6-35B-A3B-abliterated.i1-Q3_K_M.gguf',
    [string]$GuideLLMPython = 'E:\AI\guidellm-venv\Scripts\python.exe',
    [switch]$SkipDense,
    [switch]$SkipMoe,
    [switch]$SkipLong,
    [switch]$Quick
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $Dense) { $Dense = Join-Path $root 'models\JOSIEFIED-Qwen3-8B-Q4_K_M.gguf' }
$ghost = Join-Path $root 'target\release\ghost-infer.exe'
foreach ($f in @($ghost, $Dense, $Long, $GuideLLMPython) + $Moe) { if (-not (Test-Path $f)) { throw "not found: $f" } }
# the engine (serve) or the app hold the GPU and RAM; a `ghost-infer monitor` window is fine
$busy = Get-CimInstance Win32_Process -Filter "Name='ghost-infer.exe' OR Name='ghost-app.exe'" | Where-Object { $_.Name -eq 'ghost-app.exe' -or $_.CommandLine -notmatch ' monitor' }
if ($busy) { throw 'the ghost-infer engine or the app is running: close them first (they hold the GPU and RAM).' }
$out = Join-Path $root ('bench-ghost-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $out | Out-Null
function Say($text) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $text" }
$inv = [Globalization.CultureInfo]::InvariantCulture
function F1($x) { [string]::Format($inv, '{0:0.0}', [double]$x) }

$os = Get-CimInstance Win32_OperatingSystem
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) -join ' '
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
$commit = (& git -C $root log --oneline -1) -join ''
$md = @('# ghost-infer benchmark', '',
        "ghost-infer $commit. $gpu; $cpu; $ram GB of RAM, $(F1 ($os.FreePhysicalMemory / 1MB)) GB free at the start. $(Get-Date -Format 'yyyy-MM-dd HH:mm').", '')

# ---- 1. dense: GuideLLM through the server, ghost-infer alone
if (-not $SkipDense) {
    Say 'dense: Qwen3 8B through the server with GuideLLM'
    $a = @{ Model = $Dense; Tokenizer = 'Qwen/Qwen3-8B'; GuideLLMPython = $GuideLLMPython; Engines = @('ghost-infer'); Out = (Join-Path $out 'dense') }
    if ($Quick) { $a.Rounds = 1; $a.Quick = $true }
    $dense = & (Join-Path $PSScriptRoot 'bench-engines.ps1') @a
    $md += '## Dense model in VRAM', '', ($dense | Out-String).Trim(), ''
}

# ---- 2. mixture of experts: the seven prompts, one model at a time (bench-readme.ps1 does the server and prompts)
if (-not $SkipMoe) {
    foreach ($m in $Moe) {
        Say "MoE: $(Split-Path $m -Leaf)"
        $a = @{ SkipDense = $true; Moe = $m; Engines = @('ghost-infer') }
        if ($Quick) { $a.Quick = $true }
        & (Join-Path $PSScriptRoot 'bench-readme.ps1') @a | Out-Null
        # bench-readme writes its own bench-readme-<time>\ folder: take its table and move the folder in here
        $run = Get-ChildItem $root -Directory -Filter 'bench-readme-*' | Sort-Object LastWriteTime | Select-Object -Last 1
        $md += (Get-Content (Join-Path $run.FullName 'moe.md') -Encoding UTF8), ''
        Move-Item $run.FullName (Join-Path $out ('moe-' + [IO.Path]::GetFileNameWithoutExtension($m)))
    }
}

# ---- 3. long contexts: ghost-infer bench with a 64K and a 128K prompt (the app's 4-bit KV cache)
if (-not $SkipLong) {
    $md += "## Long contexts: $(Split-Path $Long -Leaf)", '', '| Prompt | Reading it | Writing after it |', '|---|---|---|'
    $lens = if ($Quick) { @(4096) } else { @(65536, 131072) }
    foreach ($n in $lens) {
        Say "long: a $n-token prompt"
        $env:GHOST_INFER_BENCH_ROUNDS = '1'
        $log = Join-Path $out "long-$n.txt"
        $ErrorActionPreference = 'Continue'
        & $ghost bench $Long --gpu --kv q4 --ctx ($n + 1024) --prompt-len $n --max 128 *> $log
        $ErrorActionPreference = 'Stop'
        Remove-Item Env:GHOST_INFER_BENCH_ROUNDS
        $line = Select-String -Path $log -Pattern 'prompt (\d+) tok:\s+([\d.]+) tok/s \| generate (\d+) tok:\s+([\d.]+) tok/s' | Select-Object -Last 1
        if (-not $line) { $md += "| $n tokens | failed (see long-$n.txt) | |"; continue }
        $g = $line.Matches[0].Groups
        $pp = [double]::Parse($g[2].Value, $inv); $tg = [double]::Parse($g[4].Value, $inv)
        $secs = $n / $pp
        $read = if ($secs -ge 60) { "$(F1 ($secs / 60)) min ($([int]$pp) tokens/s)" } else { "$(F1 $secs) s ($([int]$pp) tokens/s)" }
        $md += "| $($n / 1024)K tokens | $read | **$(F1 $tg)** tokens/s |"
    }
    $md += ''
}

$md | Out-File -Encoding utf8 (Join-Path $out 'results.md')
Say "done: $out\results.md"
$md | Write-Host
