# Security policy

## Supported versions

ghost-infer has no releases yet. Security fixes go into the `main` branch; please test against the latest
commit before reporting.

## Reporting a vulnerability

Please **don't open a public issue** for a security problem. Report it privately through GitHub:
[Security → Report a vulnerability](https://github.com/affifuddin-hazam/ghost-infer/security/advisories/new).

Include what you can of:

- what the problem is and what an attacker could do with it,
- the commit you tested, and how ghost-infer was started (command line, `--host`),
- steps or a file to reproduce it (for a crafted GGUF file, how it was made rather than a large upload).

You'll get a reply within 7 days. Once a fix is ready we'll agree on when to publish, and you'll be
credited in the advisory unless you'd rather not be.

## What counts

ghost-infer runs models and serves an HTTP API on the user's own machine. Things we want to hear about:

- **The server**: anything reachable over HTTP, other than the agent's tools used as designed (below),
  that reads or writes files outside the model folders, runs code, crashes the server or uses unbounded
  memory from a request, or lets a web page on another origin drive the API from the user's browser.
- **The agent** (the tools the desktop app and the chat page give the model): its endpoints answering
  anything but the app or chat page on this computer (they need a connection from this computer, a
  localhost `Host` and the token written into the chat page); a file tool reaching outside the chat's
  folder and the folders the user added; a tool that needs the user's OK running without one (an answer
  to its request, its group set to always allowed, or a permission mode that lets it through), including
  in a scheduled task or loop; a change made in plan mode.
- **Model files**: a GGUF file that makes ghost-infer read or write out of bounds, or run code, rather
  than failing with an error.
- **The chat page and the app**: model output or a conversation that runs script in the page, or makes
  the app run something.

Out of scope:

- The server has no authentication. `--host 0.0.0.0` exposes it to everyone on the network by design;
  only use it on a network you trust.
- The `get_system_info` tool reports this computer's hardware, drives and running programs to the model
  on purpose (it can be turned off in the chat page's settings).
- What a model says. Harmful or wrong model output is a property of the model, not of ghost-infer.
- What a model does with tools the user allowed. A group set to always allowed, or a permission mode that
  lets calls through (Auto, Accept edits, Bypass permissions), is used with no further question, also by
  scheduled tasks and loops while nobody is watching (they use only what is always allowed), and what the
  model reads (web pages, files, MCP results) can steer it. Only allow always what you would let it do
  unwatched.
- MCP servers and skills the user installs: an MCP server is a program that runs as the user, and a
  skill's instructions and files go to the model. Install only ones you trust.
- Slow requests from a client you already let in (requests run one at a time).
