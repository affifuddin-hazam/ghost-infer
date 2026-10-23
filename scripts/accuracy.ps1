# How far are the fast (int8-activation) logprobs from the exact f32 maths, per prompt length?
# -Gpu measures the GPU backend instead of the CPU fast path (the reference is always exact CPU maths).
param([switch]$Gpu)
$root  = Split-Path $PSScriptRoot
$exe   = Join-Path $root 'target\release\betterllama.exe'
$model = Join-Path $root 'models\SmolLM2-135M-Instruct-Q8_0.gguf'
$text = "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg developed a system of movable metal type in Mainz. His innovation made books far cheaper to produce, and within fifty years printing shops had spread across Europe. Scholars, merchants and religious reformers all used the new technology to share ideas faster than ever before, and literacy rates slowly began to rise as printed material became common in cities and towns. The quick brown fox"
$words = $text -split ' '
function Top($prefix, [switch]$OnGpu) {
  $out = if ($OnGpu) { & $exe top $model $prefix --gpu 2>$null } else { & $exe top $model $prefix 2>$null }
  $l = $out[1].Trim() -split '\s+'; @{ id = $l[0]; lp = [double]$l[3] }
}
$worst = 0.0
foreach ($c in 4, 20, 40, 60, $words.Count) {
  $p = ($words[0..($c - 1)]) -join ' '
  $fast = Top $p -OnGpu:$Gpu
  $env:BETTERLLAMA_EXACT = '1'; $exact = Top $p; Remove-Item Env:\BETTERLLAMA_EXACT
  $d = [Math]::Abs($fast.lp - $exact.lp); $worst = [Math]::Max($worst, $d)
  "{0,3} words  exact {1,8:N4} (id {2,5})  fast {3,8:N4} (id {4,5})  diff {5:N4}" -f $c, $exact.lp, $exact.id, $fast.lp, $fast.id, $d
}
"max |fast - exact| = $([Math]::Round($worst, 4))"
