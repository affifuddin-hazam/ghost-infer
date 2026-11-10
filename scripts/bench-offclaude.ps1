# The numbers from the 2026-11-10 engine work, measured again with nothing else running (Claude closed):
#
#   1. Qwen3.6-35B-A3B (bigger than the GPU and than free RAM) through `ghost-infer bench`, each setting run
#      several times, interleaved so that RAM and disk-cache warmth do not favour one:
#        default    the expert cache admits by frequency, RAM floor off (what the CLI does by default)
#        lru        the old cache (GHOST_INFER_CACHE_LRU=1), for the before/after
#        floor2     --ram-floor 2 (the server's and the app's default)
#        gpu-only   GHOST_INFER_MOE_GPU_ONLY=1 (the CPU does no expert maths)
#      Per run: each round's tokens/s (round 1 is the first reply after loading), the prompt's speed, the
#      process's CPU time, the disk paging while it ran and the lowest available RAM.
#   2. Drafting by lookup on a dense model (Qwen3-4B): one copy-heavy request (rename through a file) and one
#      new-text request through the server, with and without lookup; the replies must be the same text.
#      Then the full set, scripts\bench-lookup.ps1 (11 prompts, 4B and 8B), ghost-infer with and without lookup.
#   3. Long contexts on Qwen3.6-35B-A3B through `ghost-infer bench` (--no-lookup: its prompt is one sentence
#      repeated, which lookup would copy): prompt reading and writing after 16K, 64K and 128K tokens, now and
#      with the attention from before 0.2.4 (GHOST_INFER_OLD_WIDE_ATT=1 GHOST_INFER_NO_TC_ATT=1; its prompt
#      kernel is attention_wide_g, ~1.25x 0.2.3's). The old 128K prompt alone takes ~27 minutes.
#   4. A draft model: Qwen3 8B with and without Qwen3-0.6B guessing (--no-lookup), the three prompts of
#      docs/benchmarks.md through the server; the replies must be the same text.
#
# Close Claude, browsers and anything else that holds RAM or GPU time, then from a plain PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\bench-offclaude.ps1
#   ... -Repeats 2                (runs per setting; default 3)
#   ... -Settings default,lru     (only these)
#   ... -SkipLookup / -SkipMoe / -SkipLong / -SkipDraft
#   ... -LongLens 16384,65536     (contexts for part 3; default 16384,65536,131072)
#   ... -WaitSeconds 60           (start, then close what you need to; it waits before measuring)
#   ... -Quick                    (a smoke test: minutes, numbers mean nothing)
#
# Everything goes to bench-offclaude-<time>\ (gitignored): raw outputs and results.md.
param(
    [string]$Moe = 'C:\models\Huihui-Qwen3.6-35B-A3B-abliterated.i1-Q3_K_M.gguf',
    [string]$Dense = 'C:\models\Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf',
    [string[]]$Settings = @('default', 'lru', 'floor2', 'gpu-only'),
    [int]$Repeats = 3,
    [int]$Generate = 384,
    [int]$PromptLen = 512,
    [int]$Port = 8097,
    [int]$WaitSeconds = 0,
    [string]$DraftTarget = '',  # default: models\JOSIEFIED-Qwen3-8B-Q4_K_M.gguf
    [string]$DraftModel = '',   # default: models\JOSIEFIED-Qwen3-0.6B-Q8_0.gguf
    [int[]]$LongLens = @(16384, 65536, 131072),
    [switch]$SkipMoe,
    [switch]$SkipLookup,
    [switch]$SkipLong,
    [switch]$SkipDraft,
    [switch]$Quick
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$ghost = Join-Path $root 'target\release\ghost-infer.exe'
$Settings = @($Settings | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if (-not $DraftTarget) { $DraftTarget = Join-Path $root 'models\JOSIEFIED-Qwen3-8B-Q4_K_M.gguf' }
if (-not $DraftModel) { $DraftModel = Join-Path $root 'models\JOSIEFIED-Qwen3-0.6B-Q8_0.gguf' }
if ($Quick) { $Repeats = 1; $Generate = 48; $PromptLen = 64; $LongLens = @(2048) }
$needs = @($ghost) + $(if (-not $SkipMoe -or -not $SkipLong) { @($Moe) }) + $(if (-not $SkipLookup) { @($Dense) }) +
    $(if (-not $SkipDraft) { @($DraftTarget, $DraftModel) })
foreach ($f in $needs) {
    if (-not (Test-Path $f)) { throw "not found: $f (build first: cargo build --release)" }
}
$out = Join-Path $root ('bench-offclaude-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $out | Out-Null
$inv = [Globalization.CultureInfo]::InvariantCulture
function Say($text) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $text" }
function Median($xs) {
    $s = @($xs | Where-Object { $_ -ne $null } | Sort-Object)
    if (-not $s.Count) { return $null }
    if ($s.Count % 2) { return $s[[int][math]::Floor($s.Count / 2)] }
    return ($s[$s.Count / 2 - 1] + $s[$s.Count / 2]) / 2
}
function Fmt($x, $digits = 1) { if ($x -eq $null) { '-' } else { ([double]$x).ToString("F$digits", $inv) } }

# ---- what else is running
$busy = @(Get-Process claude*, chrome, msedge, firefox, brave, 'Docker Desktop', vmmem -ErrorAction SilentlyContinue | Group-Object ProcessName | ForEach-Object { $_.Name })
if ($busy.Count) { Say "still running: $($busy -join ', ') (they take RAM and CPU: the MoE numbers depend on it)" }
if (Get-Process ghost-infer -ErrorAction SilentlyContinue) { throw 'ghost-infer is already running (the app or a server): quit it first.' }
if ($WaitSeconds -gt 0) { Say "waiting $WaitSeconds s"; Start-Sleep -Seconds $WaitSeconds }
$avail0 = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
$gpu = (& nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>$null) -join ''
Say "available RAM $avail0 MB; GPU: $gpu"

$envNames = @('GHOST_INFER_CACHE_LRU', 'GHOST_INFER_MOE_GPU_ONLY', 'GHOST_INFER_MOE_STATS', 'GHOST_INFER_RAM_FLOOR', 'GHOST_INFER_NO_LOOKUP',
    'GHOST_INFER_OLD_WIDE_ATT', 'GHOST_INFER_NO_TC_ATT', 'GHOST_INFER_BENCH_ROUNDS')
function Clear-GhostEnv { foreach ($n in $envNames) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue } }

# ---- 1. the mixture of experts
$moeRows = @()
if (-not $SkipMoe) {
    $configs = @{
        'default'  = @{ args = @('--ram-floor', '0'); env = @{} }
        'lru'      = @{ args = @('--ram-floor', '0'); env = @{ GHOST_INFER_CACHE_LRU = '1' } }
        'floor2'   = @{ args = @('--ram-floor', '2'); env = @{} }
        'gpu-only' = @{ args = @('--ram-floor', '0'); env = @{ GHOST_INFER_MOE_GPU_ONLY = '1' } }
    }
    foreach ($s in $Settings) { if (-not $configs.ContainsKey($s)) { throw "unknown setting $s (default, lru, floor2, gpu-only)" } }
    for ($r = 1; $r -le $Repeats; $r++) {
        foreach ($s in $Settings) {
            $cfg = $configs[$s]
            Clear-GhostEnv
            $env:GHOST_INFER_MOE_STATS = '1'
            foreach ($k in $cfg.env.Keys) { Set-Item "Env:$k" $cfg.env[$k] }
            $log = Join-Path $out "moe-$s-$r"
            $argList = @('bench', $Moe, '-n', "$Generate", '--prompt-len', "$PromptLen", '--gpu') + $cfg.args
            Say "35B $s, run $r of $Repeats"
            $p = Start-Process -FilePath $ghost -ArgumentList $argList -RedirectStandardOutput "$log.out.txt" -RedirectStandardError "$log.err.txt" -PassThru -NoNewWindow
            $pages = @(); $minAvail = 999999; $cpuSec = 0; $t0 = Get-Date; $wall = 0
            while (-not $p.HasExited) {
                $c = Get-Counter '\Memory\Pages Input/sec', '\Memory\Available MBytes' -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue
                if ($c) { $pages += $c.CounterSamples[0].CookedValue; $minAvail = [math]::Min($minAvail, $c.CounterSamples[1].CookedValue) }
                # Windows PowerShell loses an exited process's CPU time: read it while it runs
                try { $cpuSec = (Get-Process -Id $p.Id -ErrorAction Stop).TotalProcessorTime.TotalSeconds; $wall = ((Get-Date) - $t0).TotalSeconds } catch {}
            }
            $p.WaitForExit()
            $rounds = @(Get-Content "$log.err.txt" | Select-String '^round (\d+): ([\d.]+) tok/s' | ForEach-Object { [double]::Parse($_.Matches[0].Groups[2].Value, $inv) })
            $hits = @(Get-Content "$log.err.txt" | Select-String '([\d.]+)% of expert picks' | ForEach-Object { [double]::Parse($_.Matches[0].Groups[1].Value, $inv) })
            $summary = (Get-Content "$log.out.txt" | Select-String 'prompt \d+ tok: +([\d.]+) tok/s \| generate \d+ tok: +([\d.]+) tok/s')
            $promptTps = if ($summary) { [double]::Parse($summary.Matches[0].Groups[1].Value, $inv) } else { $null }
            $row = [pscustomobject]@{
                setting = $s; run = $r; rounds = $rounds; prompt = $promptTps; hits = $hits
                cpu = if ($wall -gt 0) { 100 * $cpuSec / ($wall * [Environment]::ProcessorCount) } else { $null }
                disk = if ($pages.Count) { ($pages | Measure-Object -Average).Average * 4 / 1024 } else { $null }
                avail = $minAvail; exit = $p.ExitCode
            }
            $moeRows += $row
            Say ("  rounds {0} | prompt {1} tok/s | CPU {2}% | disk {3} MB/s | lowest available {4} MB" -f (($rounds | ForEach-Object { Fmt $_ }) -join ' / '), (Fmt $promptTps), (Fmt $row.cpu 0), (Fmt $row.disk 0), $minAvail)
        }
    }
    Clear-GhostEnv
}

# ---- 2. lookup on a dense model, through the server
$lookupRows = @()
if (-not $SkipLookup) {
    $sampler = Get-Content (Join-Path $root 'src\sampler.rs') -Raw
    $sampler = $sampler.Substring(0, [math]::Min(5000, $sampler.Length))
    $requests = @(
        @{ name = 'rename through a file (copy-heavy)'; text = "Rename ``cand`` to ``candidates`` everywhere in this file. Reply with the whole file in one code block and nothing else.`n`n``````rust`n$sampler`n``````" },
        @{ name = 'explain (new text)'; text = 'Explain in about 300 words how a hash map works and why lookups are fast on average.' }
    )
    $maxTokens = if ($Quick) { 64 } else { 600 }
    foreach ($mode in @('lookup', 'no-lookup')) {
        Clear-GhostEnv
        $log = Join-Path $out "lookup-$mode"
        $argList = @('serve', $Dense, '--gpu', '--port', "$Port") + $(if ($mode -eq 'no-lookup') { @('--no-lookup') } else { @() })
        Say "4B server, $mode"
        $srv = Start-Process -FilePath $ghost -ArgumentList $argList -RedirectStandardOutput "$log.out.txt" -RedirectStandardError "$log.err.txt" -PassThru -NoNewWindow
        try {
            $ready = $false
            for ($i = 0; $i -lt 120 -and -not $ready; $i++) {
                Start-Sleep -Seconds 1
                try { Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2 | Out-Null; $ready = $true } catch {}
            }
            if (-not $ready) { throw "the 4B server did not start (see $log.err.txt)" }
            foreach ($q in @($requests) + @($requests)) {  # each twice: the first warms up, the second counts
                $body = @{ model = 'm'; messages = @(@{ role = 'user'; content = $q.text }); max_tokens = $maxTokens; temperature = 0 } | ConvertTo-Json -Depth 5
                $resp = Invoke-RestMethod -Method Post "http://127.0.0.1:$Port/v1/chat/completions" -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 900
                $line = @(Get-Content "$log.err.txt" | Select-String 'generated in [\d.]+s \(([\d.]+) tok/s\)')[-1]
                $tps = if ($line) { [double]::Parse($line.Matches[0].Groups[1].Value, $inv) } else { $null }
                $lookupRows += [pscustomobject]@{ mode = $mode; name = $q.name; tps = $tps; text = $resp.choices[0].message.content; tokens = $resp.usage.completion_tokens }
            }
        } finally {
            Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
    }
    # the full set: 11 prompts on the 4B and the 8B, with and without lookup (it writes bench-lookup-<time>\)
    Clear-GhostEnv
    Say 'the full lookup set (scripts\bench-lookup.ps1)'
    $before = @(Get-ChildItem $root -Directory -Filter 'bench-lookup-*' | ForEach-Object FullName)
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'bench-lookup.ps1') -Engines 'ghost-infer,ghost-infer-no-lookup' $(if ($Quick) { '-Quick' })
    $lookupSet = @(Get-ChildItem $root -Directory -Filter 'bench-lookup-*' | Where-Object { $before -notcontains $_.FullName } | Sort-Object LastWriteTime)[-1]
    if ($lookupSet -and (Test-Path (Join-Path $lookupSet.FullName 'results.md'))) {
        Copy-Item (Join-Path $lookupSet.FullName 'results.md') (Join-Path $out 'lookup-set.md')
    } else { Say '  the lookup set wrote no results.md' }
}

# ---- 3. long contexts on the 35B: now, and with the attention from before 0.2.4
$longRows = @()
if (-not $SkipLong) {
    $kinds = [ordered]@{ 'now' = @{}; 'before 0.2.4' = @{ GHOST_INFER_OLD_WIDE_ATT = '1'; GHOST_INFER_NO_TC_ATT = '1' } }
    foreach ($len in $LongLens) {
        foreach ($kind in $kinds.Keys) {
            Clear-GhostEnv
            # bench keeps the best round. Two rounds, so the second runs with the experts already in the GPU's cache
            # (a single round started cold: 16 tok/s at 2K); one at 128K, where reading the prompt warms them
            # and the old way reads it in ~27 minutes
            $env:GHOST_INFER_BENCH_ROUNDS = if ($len -ge 100000) { '1' } else { '2' }
            foreach ($k in $kinds[$kind].Keys) { Set-Item "Env:$k" $kinds[$kind][$k] }
            $log = Join-Path $out ("long-{0}-{1}" -f $len, ($kind -replace '[^a-z0-9]', ''))
            Say "35B long context: $len tokens, $kind"
            $t0 = Get-Date
            $p = Start-Process -FilePath $ghost -ArgumentList @('bench', $Moe, '--gpu', '--no-lookup', '--prompt-len', "$len", '-n', '128') `
                -RedirectStandardOutput "$log.out.txt" -RedirectStandardError "$log.err.txt" -PassThru -NoNewWindow
            $p.WaitForExit()
            $m = Get-Content "$log.out.txt" | Select-String 'prompt \d+ tok: +([\d.]+) tok/s \| generate \d+ tok: +([\d.]+) tok/s'
            $hash = Get-Content "$log.err.txt" | Select-String 'tokens ([0-9a-f]{16})' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Last 1
            $row = [pscustomobject]@{
                len = $len; kind = $kind; exit = $p.ExitCode; hash = $hash; minutes = ((Get-Date) - $t0).TotalMinutes
                prompt = if ($m) { [double]::Parse($m.Matches[0].Groups[1].Value, $inv) } else { $null }
                gen = if ($m) { [double]::Parse($m.Matches[0].Groups[2].Value, $inv) } else { $null }
            }
            $longRows += $row
            Say ("  prompt {0} tok/s ({1} min to read) | writing {2} tok/s" -f (Fmt $row.prompt 0), (Fmt ($(if ($row.prompt) { $len / $row.prompt / 60 } else { $null })) 1), (Fmt $row.gen))
        }
    }
    Clear-GhostEnv
}

# ---- 4. a draft model: the 8B with and without the 0.6B guessing, through the server
$draftRows = @()
if (-not $SkipDraft) {
    $prompts = @(
        'Write a Python function that checks whether a number is prime, with a short docstring and two examples.',
        'Name three effects of the printing press.',
        'Explain in simple words why the sky is blue.'
    )
    $maxTokens = if ($Quick) { 48 } else { 256 }
    foreach ($mode in @('draft', 'plain')) {
        Clear-GhostEnv
        $log = Join-Path $out "draft-$mode"
        $argList = @('serve', $DraftTarget, '--gpu', '--no-lookup', '--port', "$Port") + $(if ($mode -eq 'draft') { @('--draft', $DraftModel) } else { @() })
        Say "8B server, $mode"
        $srv = Start-Process -FilePath $ghost -ArgumentList $argList -RedirectStandardOutput "$log.out.txt" -RedirectStandardError "$log.err.txt" -PassThru -NoNewWindow
        try {
            $ready = $false
            for ($i = 0; $i -lt 120 -and -not $ready; $i++) {
                Start-Sleep -Seconds 1
                try { Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2 | Out-Null; $ready = $true } catch {}
            }
            if (-not $ready) { throw "the 8B server did not start (see $log.err.txt)" }
            foreach ($text in @($prompts) + @($prompts)) {  # each twice: the first warms up, the second counts
                $body = @{ model = 'm'; messages = @(@{ role = 'user'; content = $text }); max_tokens = $maxTokens; temperature = 0 } | ConvertTo-Json -Depth 5
                $resp = Invoke-RestMethod -Method Post "http://127.0.0.1:$Port/v1/chat/completions" -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 900
                $line = @(Get-Content "$log.err.txt" | Select-String 'generated in [\d.]+s \(([\d.]+) tok/s\)')[-1]
                $tps = if ($line) { [double]::Parse($line.Matches[0].Groups[1].Value, $inv) } else { $null }
                $draftRows += [pscustomobject]@{ mode = $mode; name = $text; tps = $tps; text = $resp.choices[0].message.content }
            }
        } finally {
            Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
    }
}

# ---- results.md
$md = @("# ghost-infer, measured with Claude closed", "", "$(Get-Date -Format 'yyyy-MM-dd HH:mm'), commit $(git -C $root rev-parse --short HEAD 2>$null), available RAM at the start $avail0 MB, GPU $gpu", "")
if ($moeRows.Count) {
    $md += "## Qwen3.6-35B-A3B ($PromptLen-token prompt, $Generate generated, $Repeats runs each, medians)", ""
    $md += '| setting | first reply after load | 2nd | 3rd | prompt | expert picks on the GPU (gpu-only: all) | CPU (all threads) | disk reads | lowest available RAM |'
    $md += '|---|---|---|---|---|---|---|---|---|'
    foreach ($s in $Settings) {
        $rows = @($moeRows | Where-Object { $_.setting -eq $s })
        $col = { param($i) Median ($rows | ForEach-Object { if ($_.rounds.Count -gt $i) { $_.rounds[$i] } }) }
        $md += "| $s | $(Fmt (& $col 0)) | $(Fmt (& $col 1)) | $(Fmt (& $col 2)) | $(Fmt (Median ($rows | ForEach-Object { $_.prompt })) 0) | $(Fmt (Median ($rows | ForEach-Object { $_.hits[-1] })))% | $(Fmt (Median ($rows | ForEach-Object { $_.cpu })) 0)% | $(Fmt (Median ($rows | ForEach-Object { $_.disk })) 0) MB/s | $(Fmt (Median ($rows | ForEach-Object { $_.avail })) 0) MB |"
    }
    $md += "", "Every run (tokens/s per round):", ""
    foreach ($row in $moeRows) { $md += "- $($row.setting) run $($row.run): $(($row.rounds | ForEach-Object { Fmt $_ }) -join ' / ')$(if ($row.exit) { " (exit code $($row.exit))" })" }
    $md += ""
}
if ($lookupRows.Count) {
    $md += "## Lookup on Qwen3-4B (server, greedy; the second of two identical requests counts)", ""
    $md += '| request | with lookup | without | same reply |', '|---|---|---|---|'
    foreach ($name in ($lookupRows | ForEach-Object { $_.name } | Select-Object -Unique)) {
        $on = @($lookupRows | Where-Object { $_.mode -eq 'lookup' -and $_.name -eq $name })[-1]
        $off = @($lookupRows | Where-Object { $_.mode -eq 'no-lookup' -and $_.name -eq $name })[-1]
        $md += "| $name | $(Fmt $on.tps) tok/s | $(Fmt $off.tps) tok/s | $(if ($on.text -ceq $off.text) { 'yes' } else { '**no**' }) |"
    }
    $md += ""
}
if (Test-Path (Join-Path $out 'lookup-set.md')) {
    $md += "## The full lookup set (bench-lookup.ps1)", ""
    $md += @(Get-Content (Join-Path $out 'lookup-set.md') -Encoding UTF8 | Select-Object -Skip 1 | ForEach-Object { $_ -replace '^## ', '### ' })
    $md += ""
}
if ($longRows.Count) {
    $md += "## Long contexts on Qwen3.6-35B-A3B (ghost-infer bench --no-lookup, 128 generated; best of 2 rounds, 1 at 128K)", ""
    $md += "before 0.2.4: GHOST_INFER_OLD_WIDE_ATT=1 GHOST_INFER_NO_TC_ATT=1 (its prompt kernel is ~1.25x 0.2.3's).", ""
    $md += '| context | prompt reading, now | before 0.2.4 | minutes to read, now | before | writing after it, now | before | same tokens |'
    $md += '|---|---|---|---|---|---|---|---|'
    foreach ($len in $LongLens) {
        $n = @($longRows | Where-Object { $_.len -eq $len -and $_.kind -eq 'now' })[0]
        $b = @($longRows | Where-Object { $_.len -eq $len -and $_.kind -ne 'now' })[0]
        $mins = { param($r) if ($r.prompt) { $len / $r.prompt / 60 } else { $null } }
        $md += "| $len | $(Fmt $n.prompt 0) tok/s | $(Fmt $b.prompt 0) tok/s | $(Fmt (& $mins $n) 1) | $(Fmt (& $mins $b) 1) | $(Fmt $n.gen) tok/s | $(Fmt $b.gen) tok/s | $(if ($n.hash -and $n.hash -eq $b.hash) { 'yes' } else { 'no' }) |"
    }
    $md += ""
}
if ($draftRows.Count) {
    $md += "## Draft model: $(Split-Path $DraftTarget -Leaf) with $(Split-Path $DraftModel -Leaf) (server, greedy, --no-lookup; the second of two identical requests counts)", ""
    $md += '| prompt | with the draft | without | same reply |', '|---|---|---|---|'
    foreach ($name in ($draftRows | ForEach-Object { $_.name } | Select-Object -Unique)) {
        $on = @($draftRows | Where-Object { $_.mode -eq 'draft' -and $_.name -eq $name })[-1]
        $off = @($draftRows | Where-Object { $_.mode -eq 'plain' -and $_.name -eq $name })[-1]
        $md += "| $name | $(Fmt $on.tps) tok/s | $(Fmt $off.tps) tok/s | $(if ($on.text -ceq $off.text) { 'yes' } else { '**no**' }) |"
    }
    $md += ""
}
$md -join "`n" | Set-Content -Encoding utf8 (Join-Path $out 'results.md')
Say "done: $(Join-Path $out 'results.md')"
Get-Content (Join-Path $out 'results.md')
