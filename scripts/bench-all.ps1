# Every README benchmark in one unattended run, all engines on this computer as it is now (ghost-infer, llama.cpp
# b11747, Ollama 0.40.13), one after another; keeps the PC awake until it is done. About 2-3 hours.
#
#   1. bench-readme.ps1 on Qwen3.6-35B-A3B: the dense table (Qwen3 8B, GuideLLM, every engine) and the MoE table;
#   2. bench-readme.ps1 -SkipDense on Qwen3-30B-A3B: the MoE table;
#   3. bench-ghost.ps1 -SkipDense -SkipMoe: ghost-infer's long contexts (64K and 128K prompts).
#
# Close the ghost app, the engine and Ollama (its tray icon: Quit) first, then:
#   powershell -ExecutionPolicy Bypass -File scripts\bench-all.ps1
# A part that fails is noted and the next one runs. The summary goes to bench-all-<time>.txt (gitignored).
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$log = Join-Path $root ('bench-all-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.txt')
function Say($text) { $line = "[$((Get-Date).ToString('HH:mm:ss'))] $text"; Write-Host $line; Add-Content -Path $log -Value $line }

# no sleep while it runs (the display may still turn off)
Add-Type -Namespace Ghost -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[Ghost.Power]::SetThreadExecutionState(0x80000001) | Out-Null # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$parts = @(
    @{ Name = 'Qwen3.6-35B-A3B + dense'; Script = 'bench-readme.ps1'
       Args = @{ Moe = 'D:\models\Huihui-Qwen3.6-35B-A3B-abliterated.i1-Q3_K_M.gguf'; Engines = @('ghost-infer', 'llama.cpp', 'ollama'); OllamaLinkStore = 'D:\ghost-bench-ollama' } },
    @{ Name = 'Qwen3-30B-A3B'; Script = 'bench-readme.ps1'
       Args = @{ SkipDense = $true; Moe = (Join-Path $root 'models\Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf'); Engines = @('ghost-infer', 'llama.cpp', 'ollama'); OllamaLinkStore = 'E:\ghost-bench-ollama' } },
    @{ Name = 'ghost-infer long contexts'; Script = 'bench-ghost.ps1'; Args = @{ SkipDense = $true; SkipMoe = $true } }
)
Say "bench-all: $($parts.Count) parts; RAM $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)) GB"
try {
    foreach ($p in $parts) {
        Say "start: $($p.Name)"
        $t = Get-Date
        try {
            $a = $p.Args
            & (Join-Path $PSScriptRoot $p.Script) @a 2>&1 | Out-Host
            Say "done: $($p.Name) in $([int]((Get-Date) - $t).TotalMinutes) min"
        } catch {
            Say "FAILED: $($p.Name): $($_.Exception.Message)"
        }
        # its results folder (the newest bench-readme-* / bench-ghost-*)
        $dir = Get-ChildItem $root -Directory | Where-Object { $_.Name -match '^bench-(readme|ghost)-' } | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($dir) { Say "   results: $($dir.FullName)\results.md" }
    }
} finally {
    [Ghost.Power]::SetThreadExecutionState(0x80000000) | Out-Null # back to normal
    Say 'bench-all finished'
}
