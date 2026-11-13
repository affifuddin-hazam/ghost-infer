// Claude Code status line for claude-local: ghost-infer's generation speed (GET /api/speed), the reply being
// written or the last one. Claude Code runs this after each update and shows the first line it prints.
// Plain ASCII and short: wide characters (emoji, box lines) and lines wider than the window make the terminal
// draw the line over itself.
const http = require("http");

const GREEN = "\x1b[32m", CYAN = "\x1b[36m", DIM = "\x1b[2m", BOLD = "\x1b[1m", RESET = "\x1b[0m";

const req = http.get({ host: "127.0.0.1", port: 8080, path: "/api/speed", timeout: 800 }, (res) => {
  let body = "";
  res.on("data", (c) => (body += c));
  res.on("end", () => {
    try {
      const s = JSON.parse(body);
      const rate = s.tok_s > 0 ? `${BOLD}${GREEN}${s.tok_s.toFixed(1)} tok/s${RESET}` : `${DIM}-- tok/s${RESET}`;
      const what = s.reading_prompt ? `${CYAN}reading prompt${RESET}` : s.generating ? `${CYAN}${s.tokens} tokens${RESET}` : `${DIM}${s.tokens} tokens${RESET}`;
      console.log(`${rate} ${DIM}|${RESET} ${what} ${DIM}| Qwen3.5 35B Claude distill (abliterated), local RTX 2060 SUPER${RESET}`);
    } catch {
      console.log(`${DIM}ghost-infer: no answer${RESET}`);
    }
  });
});
req.on("timeout", () => req.destroy());
req.on("error", () => console.log(`${DIM}ghost-infer is not running${RESET}`));
