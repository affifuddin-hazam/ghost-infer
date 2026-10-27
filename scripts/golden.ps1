# Regression check: next-token log-probabilities at several prompt lengths plus one greedy reply.
# Usage: scripts\golden.ps1 [-Save]   (-Save writes the reference file, otherwise compares against it)
param([switch]$Save)
$root  = Split-Path $PSScriptRoot
$exe   = Join-Path $root 'target\release\ghost-infer.exe'
$model = Join-Path $root 'models\SmolLM2-135M-Instruct-Q8_0.gguf'
$ref   = Join-Path $root 'scripts\golden.txt'

$text = "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg developed a system of movable metal type in Mainz. His innovation made books far cheaper to produce, and within fifty years printing shops had spread across Europe. Scholars, merchants and religious reformers all used the new technology to share ideas faster than ever before, and literacy rates slowly began to rise as printed material became common in cities and towns. The quick brown fox"
$words = $text -split ' '
$lines = foreach ($c in 4, 20, 40, 60, $words.Count) {
  $top = & $exe top $model (($words[0..($c - 1)]) -join ' ') 2>$null
  # keep top-3: id and logprob
  ($top[1..3] | ForEach-Object { $f = $_.Trim() -split '\s+'; "$($f[0])=$($f[3])" }) -join ' '
}
$reply = (& $exe run $model -p 'What is the capital of France?' -t 0 -n 40 2>$null) -join ' '
$lines += "reply: $reply"

if ($Save) { $lines | Set-Content -Encoding utf8 $ref; "saved $($lines.Count) lines to $ref"; return }

$want = Get-Content -Encoding utf8 $ref
# int8 activation rounding alone moves logprobs by up to ~0.1 vs exact maths (scripts\accuracy.ps1),
# so two correct builds can differ by ~0.2; more than that means something changed for real
$tolerance = 0.2
$worst = 0.0; $ok = $true
for ($i = 0; $i -lt $want.Count - 1; $i++) {
  $a = $want[$i] -split ' '; $b = $lines[$i] -split ' '
  for ($j = 0; $j -lt $a.Count; $j++) {
    $ida, $lpa = $a[$j] -split '='; $idb, $lpb = $b[$j] -split '='
    if ($j -eq 0 -and $ida -ne $idb) { $ok = $false; "line ${i}: top token changed $ida -> $idb" }
    if ($ida -eq $idb) { $worst = [Math]::Max($worst, [Math]::Abs([double]$lpa - [double]$lpb)) }
  }
}
if ($want[-1] -ne $lines[-1]) { "greedy reply changed:`n  was: $($want[-1])`n  now: $($lines[-1])" }
$pass = $ok -and $worst -le $tolerance
"$(if ($pass) { 'PASS' } else { 'FAIL' })  top-1 tokens identical: $ok   max logprob drift: $([Math]::Round($worst, 4)) (limit $tolerance)   reply identical: $($want[-1] -eq $lines[-1])"
