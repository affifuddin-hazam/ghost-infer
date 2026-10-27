# Generation-path check at long contexts: feed a prompt token by token (GHOST_INFER_STEPWISE) on the GPU,
# which exercises the f32 matvec and the split-context (flash-decoding) attention, and compare the final
# next-token logprob with exact CPU maths.
$root  = Split-Path $PSScriptRoot
$exe   = Join-Path $root 'target\release\ghost-infer.exe'
$model = Join-Path $root 'models\SmolLM2-135M-Instruct-Q8_0.gguf'
$base = "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg developed movable metal type in Mainz, and printing soon spread across Europe. "
foreach ($reps in 1, 8, 30, 52) {
  $p = ($base * $reps) + "The printing press was invented by"
  $env:GHOST_INFER_STEPWISE = '1'; $g = & $exe top $model $p --gpu 2>$null; Remove-Item Env:\GHOST_INFER_STEPWISE
  $env:GHOST_INFER_EXACT = '1';    $e = & $exe top $model $p 2>$null;       Remove-Item Env:\GHOST_INFER_EXACT
  $ntok = ($e[0] -split ' ')[0]
  $gl = $g[1].Trim() -split '\s+'; $el = $e[1].Trim() -split '\s+'
  "{0,5} tokens  exact {1,8} (id {2,5})  gpu-stepwise {3,8} (id {4,5})  diff {5:N5}" -f $ntok, $el[3], $el[0], $gl[3], $gl[0], [Math]::Abs([double]$el[3] - [double]$gl[3])
}
