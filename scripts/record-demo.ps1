# Records the README's demo: the desktop app fixing a failing test, in real time. It runs on a demo data
# folder and a small made-up project in %TEMP%\ghost-demo, so none of your own chats, memory or folders
# show. The server warms the model up and does the task once unrecorded (so the recorded take runs at full
# speed, with the experts in memory and the prompt cached), then the app does it again while ffmpeg records
# the screen where the app is placed (kept on top, so nothing covers it).
#
# Close what you can first (the model's experts need the RAM), then:
#   powershell -ExecutionPolicy Bypass -File scripts\record-demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\record-demo.ps1 -NoDocs   (only into demo-out\)
#
# Results go to demo-out\<time>\ (the raw recording, demo.gif, demo.mp4, screenshot-app.png, the server's
# log), and into docs\ with the README's caption, unless -NoDocs. Build both programs first if the code
# changed (cargo build --release -p ghost-infer --target-dir target\glm, and -p ghost-app --target-dir
# target\app).
param(
    [string]$Model = 'C:\models\Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf',
    [string]$Server = '',   # default: target\glm\release\ghost-infer.exe
    [string]$App = '',      # default: target\app\release\ghost-app.exe
    [string]$FFmpeg = '',   # default: ffmpeg on the PATH, else ImageMagick's
    [int]$Ctx = 16384,
    [int]$Port = 8097,
    [string]$Message = 'total() in cart.py gets the discount wrong. Fix it and run the tests.',
    [int]$GifWidth = 960,
    [switch]$NoDryRun,
    [switch]$NoDocs
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $Server) { $Server = Join-Path $root 'target\glm\release\ghost-infer.exe' }
if (-not $App) { $App = Join-Path $root 'target\app\release\ghost-app.exe' }
if (-not $FFmpeg) {
    $FFmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
    if (-not $FFmpeg) { $FFmpeg = Get-ChildItem 'C:\Program Files\ImageMagick*\ffmpeg.exe' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName }
}
foreach ($f in @($Model, $Server, $App, $FFmpeg)) {
    if (-not $f -or -not (Test-Path $f)) { throw "not found: $f" }
}
$utf8 = New-Object Text.UTF8Encoding $false # no BOM: the server's JSON reader takes none
function Write-Text($path, $text) { [IO.File]::WriteAllText($path, $text, $utf8) }
function Say($text) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $text" }

# ---- the demo project and data folder

$demo = Join-Path $env:TEMP 'ghost-demo'
$cart = Join-Path $demo 'cart'
$data = Join-Path $demo 'appdata\ghost-infer'
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$out = Join-Path $root "demo-out\$stamp"
if (Test-Path $demo) { Remove-Item -Recurse -Force $demo }
New-Item -ItemType Directory -Force $cart, (Join-Path $data 'chats'), $out | Out-Null

$cartPy = @'
"""A shopping cart's totals."""


def subtotal(items):
    """The sum of price * quantity over (price, quantity) pairs."""
    return sum(price * qty for price, qty in items)


def total(items, discount_percent=0):
    """The subtotal less a percentage discount, rounded to cents."""
    s = subtotal(items)
    return round(s - discount_percent, 2)
'@
$testPy = @'
import unittest

from cart import subtotal, total


class CartTest(unittest.TestCase):
    def test_subtotal(self):
        self.assertEqual(subtotal([(2.50, 2), (1.25, 4)]), 10.0)

    def test_total_without_discount(self):
        self.assertEqual(total([(10.0, 3)]), 30.0)

    def test_total_with_discount(self):
        self.assertEqual(total([(20.0, 2)], discount_percent=25), 30.0)


if __name__ == "__main__":
    unittest.main()
'@
function Reset-Project {
    Write-Text (Join-Path $cart 'cart.py') ($cartPy -replace "`r`n", "`n")
    Write-Text (Join-Path $cart 'test_cart.py') ($testPy -replace "`r`n", "`n")
}
Reset-Project
Write-Text (Join-Path $cart 'GHOST.md') "# cart`n`nRun Python with ``py -3`` here (there is no ``python`` command). Tests: ``py -3 -m unittest -v``.`n"

$settings = @{ folders = @($cart); always = @('edit', 'cmd') }
Write-Text (Join-Path $data 'harness.json') ($settings | ConvertTo-Json -Compress)
Write-Text (Join-Path $data 'app.json') '{"theme":"dark","tools":["files","edit","cmd"],"rail":true,"load_last":false,"keep_going":true}'
Write-Text (Join-Path $data 'HEARTBEAT.md') "# Heartbeat checklist`n`n- Do the cart tests still pass?`n"
# a heartbeat and a daily task, so the rail shows what runs on its own (neither comes due during the take)
$now = [DateTimeOffset]::Now
$six = $now.Date.AddHours(18)
while ($six -le $now.DateTime -or $six.DayOfWeek -eq 'Saturday' -or $six.DayOfWeek -eq 'Sunday') { $six = $six.AddDays(1) }
$ms = { param($t) [DateTimeOffset]::new($t).ToUnixTimeMilliseconds() }
$schedules = '{"jobs":[{"id":"6a1f0","name":"Summarise the day''s commits","prompt":"Summarise today''s commits in the cart project.","chat":"","next":' + (& $ms $six) + ',"every":0,"daily":1080,"days":[1,2,3,4,5],"tools":["files","cmd"],"thinking":false,"last":0,"note":""}],' +
    '"heartbeat":{"every":60,"chat":"","tools":["files","cmd"],"thinking":false,"next":' + ($now.AddMinutes(55).ToUnixTimeMilliseconds()) + ',"last":0,"note":""}}'
Write-Text (Join-Path $data 'schedules.json') $schedules

# ---- the server

$base = "http://127.0.0.1:$Port"
$oldAppData = $env:LOCALAPPDATA
$env:LOCALAPPDATA = Join-Path $demo 'appdata' # the server and the app read and write the demo data only
$serverProc = $null; $appProc = $null; $ff = $null
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type @'
using System; using System.Text; using System.Runtime.InteropServices;
public class DemoWin {
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int w, int hh, uint f);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    delegate bool Cb(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] static extern bool EnumWindows(Cb cb, IntPtr p);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    // The app's own window: visible and titled "... ghost-infer" (the process also has winit's small
    // helper window, which is what Process.MainWindowHandle finds first)
    public static IntPtr Find(uint pid) {
        IntPtr found = IntPtr.Zero;
        EnumWindows((h, p) => {
            uint q; GetWindowThreadProcessId(h, out q);
            var t = new StringBuilder(256); GetWindowText(h, t, 256);
            if (q == pid && IsWindowVisible(h) && t.ToString().EndsWith("ghost-infer")) { found = h; return false; }
            return true;
        }, IntPtr.Zero);
        return found;
    }
}
'@

function Get-Json($path) {
    $wc = New-Object Net.WebClient
    $wc.Encoding = [Text.Encoding]::UTF8
    $wc.Headers.Add('X-Ghost-Token', $script:token)
    $wc.DownloadString("$base$path") | ConvertFrom-Json
}
function Clear-Chats {
    foreach ($c in @((Get-Json '/api/chats').chats)) {
        if ($c) { Invoke-WebRequest -UseBasicParsing -Method Delete -Uri "$base/api/chats/$($c.id)" -Headers @{ 'X-Ghost-Token' = $script:token } | Out-Null }
    }
}
# Waits until the reply is done: the model has named the chat, or (if naming fails) the last message has
# been a finished answer for a few seconds. `$tick` runs on every look (it keeps the app in place).
function Wait-Reply([int]$minutes = 8, [scriptblock]$tick = {}) {
    $deadline = (Get-Date).AddMinutes($minutes); $count = -1; $since = Get-Date
    while ((Get-Date) -lt $deadline) {
        & $tick
        $chats = @((Get-Json '/api/chats').chats)
        if ($chats.Count -gt 0 -and $chats[0]) {
            $t = [string]$chats[0].title
            if ($t -and -not $t.EndsWith([string][char]0x2026) -and $t -ne $Message) { return $t }
            $chat = Get-Json "/api/chats/$($chats[0].id)"
            $last = @($chat.messages)[-1]
            $n = @($chat.messages).Count
            if ($last.role -eq 'assistant' -and -not $last.tool_calls) {
                if ($n -ne $count) { $count = $n; $since = Get-Date }
                elseif (((Get-Date) - $since).TotalSeconds -gt 8) { return '(no title)' }
            } else { $count = -1 }
        }
        Start-Sleep -Milliseconds 500
    }
    throw "the reply did not finish in $minutes minutes"
}

try {
    Say "starting the server on $(Split-Path $Model -Leaf)"
    $serverProc = Start-Process -FilePath $Server -ArgumentList "serve `"$Model`" --gpu --ctx $Ctx --port $Port" -WindowStyle Hidden -PassThru `
        -RedirectStandardError (Join-Path $out 'server.log') -RedirectStandardOutput (Join-Path $out 'server-out.log')
    $loaded = $false
    for ($i = 0; $i -lt 300 -and -not $loaded; $i++) {
        Start-Sleep -Seconds 1
        if ($serverProc.HasExited) { throw "the server stopped; see $out\server.log" }
        try { $loaded = [bool](Invoke-RestMethod "$base/api/models").current } catch {}
    }
    if (-not $loaded) { throw 'the model did not load in 5 minutes' }
    $script:token = ([regex]'[0-9a-f]{32}').Match((New-Object Net.WebClient).DownloadString("$base/")).Value
    $sys = Invoke-RestMethod "$base/api/system"

    Say 'warming up (the first replies bring the experts into memory)'
    foreach ($q in @('Explain how unittest finds tests in a folder, briefly.', 'Write a Python function that rounds money to cents.', 'List three ways to test a Python function, one line each.')) {
        $body = @{ messages = @(@{ role = 'user'; content = $q }); max_tokens = 150 } | ConvertTo-Json -Depth 5 -Compress
        Invoke-RestMethod -Method Post -Uri "$base/v1/chat/completions" -ContentType 'application/json' -Body $body -TimeoutSec 600 | Out-Null
    }
    if (-not $NoDryRun) {
        Say 'the task once, not recorded'
        $body = @{ message = $Message; tools = @('files', 'edit', 'cmd'); thinking = $true; max_tokens = 4096; keep_going = $true } | ConvertTo-Json -Compress
        Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base/api/agent" -ContentType 'application/json' -Headers @{ 'X-Ghost-Token' = $script:token } -Body $body -TimeoutSec 600 | Out-Null
        Wait-Reply | Out-Null
        Clear-Chats
        Reset-Project
    }

    # ---- the take

    $w = 1180; $h = 780
    $screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    $x = $screen.X + [int](($screen.Width - $w) / 2); $y = $screen.Y + [int](($screen.Height - $h) / 2)
    $raw = Join-Path $out 'raw.mkv'
    $appProc = Start-Process -FilePath $App -ArgumentList "127.0.0.1:$Port --send `"$Message`"" -PassThru
    $hwnd = [IntPtr]::Zero
    for ($i = 0; $i -lt 300 -and $hwnd -eq [IntPtr]::Zero; $i++) { Start-Sleep -Milliseconds 50; $hwnd = [DemoWin]::Find([uint32]$appProc.Id) }
    if ($hwnd -eq [IntPtr]::Zero) { throw 'the app window did not open' }
    # placed on top where it is recorded, again until it stays there (the window still moves itself just
    # after it opens), and again on every look while the reply runs
    $place = { [DemoWin]::SetWindowPos($hwnd, [IntPtr](-1), $x, $y, $w, $h, 0x40) | Out-Null }
    $inPlace = 0
    for ($i = 0; $i -lt 100 -and $inPlace -lt 3; $i++) {
        & $place
        Start-Sleep -Milliseconds 100
        $r = New-Object DemoWin+RECT
        [DemoWin]::GetWindowRect($hwnd, [ref]$r) | Out-Null
        if ($r.Left -eq $x -and $r.Top -eq $y -and ($r.Right - $r.Left) -eq $w -and ($r.Bottom - $r.Top) -eq $h) { $inPlace++ } else { $inPlace = 0 }
    }
    if ($inPlace -lt 3) { throw "the app window would not stay at $x,$y ($($r.Left),$($r.Top) to $($r.Right),$($r.Bottom))" }
    [DemoWin]::SetForegroundWindow($hwnd) | Out-Null
    Say 'recording'
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FFmpeg
    $psi.Arguments = "-hide_banner -loglevel error -y -f gdigrab -framerate 15 -draw_mouse 0 -offset_x $x -offset_y $y -video_size ${w}x${h} -i desktop -c:v libx264 -preset ultrafast -qp 0 -pix_fmt yuv444p `"$raw`""
    $psi.UseShellExecute = $false; $psi.RedirectStandardInput = $true; $psi.CreateNoWindow = $true
    $ff = [System.Diagnostics.Process]::Start($psi)
    $placed = Get-Date
    $title = Wait-Reply -tick $place
    $done = Get-Date
    Start-Sleep -Seconds 5 # the title, the plan closing, the ghost's smile
    $shot = New-Object System.Drawing.Bitmap $w, $h
    [System.Drawing.Graphics]::FromImage($shot).CopyFromScreen($x, $y, 0, 0, $shot.Size)
    $shot.Save((Join-Path $out 'screenshot-app.png'), [System.Drawing.Imaging.ImageFormat]::Png)
    Start-Sleep -Seconds 1
    $ff.StandardInput.Write('q'); $ff.StandardInput.Close()
    if (-not $ff.WaitForExit(30000)) { $ff.Kill() }
    $ff = $null
    Say "recorded: `"$title`", the reply took $([int]($done - $placed).TotalSeconds) s"
} finally {
    if ($ff -and -not $ff.HasExited) { $ff.Kill() }
    foreach ($p in @($appProc, $serverProc)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force } }
    $env:LOCALAPPDATA = $oldAppData
}

# ---- the files: the whole recording (it starts with the app in place), in real time

$trim = '0.5' # ffmpeg's first frames, while it starts
$mp4 = Join-Path $out 'demo.mp4'; $gif = Join-Path $out 'demo.gif'
& $FFmpeg -hide_banner -loglevel error -y -ss $trim -i $raw -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -movflags +faststart $mp4
& $FFmpeg -hide_banner -loglevel error -y -ss $trim -i $raw -vf "fps=10,scale=${GifWidth}:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -final_delay 250 -loop 0 $gif
$seconds = [int]($done - $placed).TotalSeconds + 6
Say ("demo.gif {0:N1} MB, demo.mp4 {1:N1} MB, about {2} s, in {3}" -f ((Get-Item $gif).Length / 1MB), ((Get-Item $mp4).Length / 1MB), $seconds, $out)
Get-Content (Join-Path $out 'server.log') | Select-String 'generated' | Select-Object -Last 8 | ForEach-Object { "  $_" }

if (-not $NoDocs) {
    $docs = Join-Path $root 'docs'
    Copy-Item $gif, $mp4, (Join-Path $out 'screenshot-app.png') $docs -Force
    # the README's caption under the demo: real time now, with this model and GPU
    $label = [IO.Path]::GetFileNameWithoutExtension($Model) -replace '-(Q\d\w*)$', ' $1'
    $gpu = @($sys.gpus)[0]
    $card = if ($gpu) { "$(($gpu.name -replace '^NVIDIA ', '' -replace '^GeForce ', '')) ($([math]::Round($gpu.vram_total_gb)) GB)" } else { 'the CPU' }
    $an = if ($card -match '^[AEFHILMNORSX8]') { 'an' } else { 'a' }
    $caption = "  <br><sub>Real time, not sped up: the desktop app fixing a failing test, $label on $an $card with $([math]::Round($sys.ram_total_gb)) GB of RAM.</sub>"
    $readme = Join-Path $root 'README.md'
    $text = [IO.File]::ReadAllText($readme)
    $line = [regex]'(<img src="docs/demo\.gif"[^\n]*\n)  <br><sub>[^\n]*</sub>' # the line under the demo
    if (-not $line.IsMatch($text)) { throw 'the README has no caption under docs/demo.gif to replace' }
    $text = $line.Replace($text, { param($m) $m.Groups[1].Value + $caption }, 1)
    [IO.File]::WriteAllText($readme, $text, $utf8)
    Say 'docs\demo.gif, docs\demo.mp4, docs\screenshot-app.png and the README caption updated (git diff shows them)'
}
