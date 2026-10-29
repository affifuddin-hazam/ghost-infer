# Drafting by lookup, measured: ghost-infer with and without it, against llama.cpp with and without its own
# n-gram drafting, on copy-heavy agent work and on new text (scripts\bench-lookup.py has the prompts), for each
# model through each engine's OpenAI-compatible server, greedy, thinking off.
#
# Close everything else first (Claude, browsers: they take GPU time and RAM), then:
#   powershell -ExecutionPolicy Bypass -File scripts\bench-lookup.ps1
#   ... -Models models\a.gguf,models\b.gguf   (default: Qwen3 4B and 8B Q4_K_M from models\)
#   ... -Quick                                (a smoke test: minutes, numbers mean nothing)
#
# ghost-infer does not draft for a mixture-of-experts model (its batches run there on the CPU, other
# arithmetic than single steps), so its two columns would match for one: give dense models.
#
# Everything goes to bench-lookup-<time>\ (gitignored): the raw results and server logs per model, and
# results.md with a table per model.
param(
    [string[]]$Models = @(),
    [string]$Python = 'E:\AI\guidellm-venv\Scripts\python.exe',  # any Python 3 runs the client; this one is at hand
    [string]$LlamaDir = 'E:\AI\llama.cpp-b11747',
    [string[]]$Engines = @('ghost-infer', 'ghost-infer-no-lookup', 'llama.cpp', 'llama.cpp-ngram'),
    [string]$NgramType = 'ngram-mod',  # llama.cpp's --spec-type for the n-gram column
    [int]$Port = 8090,
    [switch]$Quick
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
# powershell -File hands "a,b" over as one string
$Models = @($Models | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Engines = @($Engines | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if (-not $Models.Count) { $Models = @('Qwen3-4B-Q4_K_M.gguf', 'JOSIEFIED-Qwen3-8B-Q4_K_M.gguf') | ForEach-Object { Join-Path $root "models\$_" } }
$ghost = Join-Path $root 'target\release\ghost-infer.exe'
$llamaServer = Join-Path $LlamaDir 'llama-server.exe'
$client = Join-Path $PSScriptRoot 'bench-lookup.py'
foreach ($f in @($ghost, $Python, $llamaServer, $client) + $Models) {
    if (-not (Test-Path $f)) { throw "not found: $f" }
}
$runArgs = if ($Quick) { @('--rounds', '1', '--warmup', '0', '--prompts', '2', '--max-tokens', '128') } else { @() }
$out = Join-Path $root ('bench-lookup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $out | Out-Null
function Say($text) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $text" }
$inv = [Globalization.CultureInfo]::InvariantCulture
function F1($x) { [string]::Format($inv, '{0:0.0}', [double]$x) }

# what the computer was doing
$os = Get-CimInstance Win32_OperatingSystem
$freeGb = F1 ($os.FreePhysicalMemory / 1MB)
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader) -join ' '
$busy = @(Get-Process claude*, chrome, msedge, firefox -ErrorAction SilentlyContinue | Group-Object ProcessName | ForEach-Object { $_.Name })
Say "free RAM $freeGb GB; GPU $gpu"
if ($busy.Count) { Say "note: also running: $($busy -join ', ') (close them for the real numbers)" }

function Wait-GpuFree([int]$belowMiB = 1500) {
    for ($i = 0; $i -lt 60; $i++) {
        $used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($used -lt $belowMiB) { Start-Sleep 3; return }
        Start-Sleep -Milliseconds 500
    }
}
function Wait-Up([string]$url, $proc, [string]$what, [string]$logs) {
    for ($i = 0; $i -lt 600; $i++) {
        if ($proc.HasExited) { throw "$what stopped while starting; see its log in $logs" }
        try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "$what did not come up in 5 minutes"
}
# stops a process this script started, with its children: only that tree
function Stop-Tree($proc) { if ($proc -and -not $proc.HasExited) { & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null; $proc.WaitForExit(10000) | Out-Null } }

# starts a server, waits until it answers, sends it the prompts, and stops it
function Measure-Server([string]$dir, [string]$label, [string]$exe, [string[]]$argv) {
    Say "   $label"
    $p = Start-Process -FilePath $exe -ArgumentList $argv -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $dir "$label.server.log") -RedirectStandardError (Join-Path $dir "$label.server.err.log")
    try {
        Wait-Up "http://127.0.0.1:$Port/health" $p $label $dir
        & $Python -I $client run --url "http://127.0.0.1:$Port" --model m --label $label --out (Join-Path $dir "$label.json") @runArgs
        if ($LASTEXITCODE -ne 0) { throw "the prompts against $label failed" }
    } finally {
        Stop-Tree $p
        Wait-GpuFree
    }
}

$titles = @{
    'ghost-infer'           = 'ghost-infer'
    'ghost-infer-no-lookup' = 'ghost-infer, --no-lookup'
    'llama.cpp'             = 'llama.cpp b11747'
    'llama.cpp-ngram'       = "llama.cpp b11747, --spec-type $NgramType"
}
$md = @('# bench-lookup results', '',
        'Through each engine''s /v1/chat/completions, streamed, greedy, thinking off, up to 512 tokens: six copy-heavy agent tasks (edit a file and reply with all of it, rename through a file, a write_file tool call carrying a file, fix and show code, convert data, quote a document) and five that write new text (a story, an explanation, a chat, new code, a word problem); one warm-up round, then two measured rounds. llama.cpp: every layer on the GPU, an 8k context, one slot.', '')
foreach ($model in $Models) {
    $name = [IO.Path]::GetFileNameWithoutExtension($model)
    $dir = Join-Path $out $name
    New-Item -ItemType Directory -Force $dir | Out-Null
    Say $name
    $labels = @()
    foreach ($e in $Engines) {
        switch ($e) {
            'ghost-infer' { Measure-Server $dir $e $ghost @('serve', "`"$model`"", '--gpu', '--port', "$Port") }
            'ghost-infer-no-lookup' { Measure-Server $dir $e $ghost @('serve', "`"$model`"", '--gpu', '--no-lookup', '--port', "$Port") }
            'llama.cpp' { Measure-Server $dir $e $llamaServer @('-m', "`"$model`"", '-ngl', '99', '-c', '8192', '-np', '1', '--cache-ram', '0', '--port', "$Port") }
            'llama.cpp-ngram' { Measure-Server $dir $e $llamaServer @('-m', "`"$model`"", '-ngl', '99', '-c', '8192', '-np', '1', '--cache-ram', '0', '--spec-type', $NgramType, '--port', "$Port") }
            default { throw "unknown engine $e" }
        }
        $labels += "$e=$($titles[$e])"
    }
    & $Python -I $client summary $dir @labels --same 'ghost-infer,ghost-infer-no-lookup' | Out-Null
    $md += "## $name", ''
    $md += Get-Content (Join-Path $dir 'summary.md') -Encoding UTF8
    # what lookup did, from ghost-infer's log
    $log = Join-Path $dir 'ghost-infer.server.err.log'
    if (Test-Path $log) {
        $kept = 0; $guessed = 0; $checks = 0
        foreach ($m in (Select-String -Path $log -Pattern 'lookup kept (\d+) of (\d+) guessed tokens in (\d+) checks')) {
            $kept += [int]$m.Matches[0].Groups[1].Value; $guessed += [int]$m.Matches[0].Groups[2].Value; $checks += [int]$m.Matches[0].Groups[3].Value
        }
        if ($guessed) { $md += '', "ghost-infer's lookup over all requests: kept $kept of $guessed guessed tokens ($([int](100 * $kept / $guessed))%) in $checks checks." }
    }
    $md += ''
}
$md += "Free RAM at the start: $freeGb GB. GPU: $gpu.$(if ($busy.Count) { " Also running: $($busy -join ', ')." })"
$md | Out-File -Encoding utf8 (Join-Path $out 'results.md')
Say "done: $out\results.md"
Get-Content (Join-Path $out 'results.md') -Encoding UTF8 | Write-Host
