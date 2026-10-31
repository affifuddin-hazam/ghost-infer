# Builds dist\GhostSetup-<version>.exe, the one-click installer (installer\ghost-infer.iss):
#   powershell -ExecutionPolicy Bypass -File scripts\build-installer.ps1
# ghost-infer and ghost-app are built for any x86-64 CPU with AVX2 and FMA (x86-64-v3: Intel Haswell, AMD Zen
# and newer), not only for this one as .cargo\config.toml has it. NVIDIA's NVRTC, the compiler ghost-infer
# builds its GPU kernels with, comes from the CUDA toolkit on this computer and ships beside them, so the
# people who install it need only their NVIDIA driver. Needs Inno Setup 6 or later (https://jrsoftware.org/isdl.php).
param(
    [string]$Iscc = '', # default: the newest Inno Setup's ISCC.exe
    [string]$Cuda = ''  # the CUDA toolkit bin folder NVRTC comes from; default: the newest 12.x
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
function Say($text) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $text" }

if (-not $Iscc) {
    # the newest Inno Setup (6 or later) where its installer puts it, for everyone or for this user
    $Iscc = @($env:ProgramFiles, ${env:ProgramFiles(x86)}, "$env:LOCALAPPDATA\Programs") |
        ForEach-Object { Get-ChildItem $_ -Directory -Filter 'Inno Setup *' -ErrorAction SilentlyContinue } |
        Where-Object { Test-Path (Join-Path $_.FullName 'ISCC.exe') } |
        Sort-Object { [int]($_.Name -replace '\D', '') } | Select-Object -Last 1 | ForEach-Object { Join-Path $_.FullName 'ISCC.exe' }
}
if (-not $Iscc -or -not (Test-Path $Iscc)) { throw 'Inno Setup not found: install it from https://jrsoftware.org/isdl.php, or pass -Iscc <path to ISCC.exe>' }
Say "packing with $Iscc"
if (-not $Cuda) {
    $Cuda = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -Filter 'v12.*' -ErrorAction SilentlyContinue |
        Sort-Object { [version]$_.Name.TrimStart('v') } | Select-Object -Last 1 | ForEach-Object { Join-Path $_.FullName 'bin' }
}
$nvrtc = Get-ChildItem $Cuda -Filter 'nvrtc64_*.dll' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.alt.dll' } | Select-Object -First 1
$builtins = Get-ChildItem $Cuda -Filter 'nvrtc-builtins64_*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $nvrtc -or -not $builtins) { throw "no NVRTC in '$Cuda' (install the CUDA toolkit 12.x, or pass -Cuda <its bin folder>)" }
$version = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches[0].Groups[1].Value

Say "building ghost-infer $version and ghost-app for x86-64-v3"
$env:RUSTFLAGS = '-C target-cpu=x86-64-v3' # takes the place of .cargo\config.toml's target-cpu=native
Push-Location $root
try {
    cargo build --release -p ghost-infer -p ghost-app --target-dir target\dist
    if ($LASTEXITCODE) { throw 'the build failed' }
} finally {
    Pop-Location
    Remove-Item Env:\RUSTFLAGS
}

$stage = Join-Path $root 'dist\stage'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
$bin = Join-Path $root 'target\dist\release'
Copy-Item (Join-Path $bin 'ghost-infer.exe'), (Join-Path $bin 'ghost-app.exe') $stage
Copy-Item $nvrtc.FullName, $builtins.FullName $stage
Copy-Item (Join-Path $root 'LICENSE-MIT'), (Join-Path $root 'LICENSE-APACHE'), (Join-Path $root 'installer\ghost.ico'), (Join-Path $root 'installer\THIRD-PARTY.txt') $stage
Say "NVRTC: $($nvrtc.Name) and $($builtins.Name) from $Cuda"

Say 'packing the installer'
& $Iscc /Q "/DVersion=$version" "/DStage=$stage" (Join-Path $root 'installer\ghost-infer.iss')
if ($LASTEXITCODE) { throw 'Inno Setup failed' }
$out = Join-Path $root "dist\GhostSetup-$version.exe"
Say ('{0} ({1:N1} MB)' -f $out, ((Get-Item $out).Length / 1MB))
