# Accessibility

ghost-infer should be usable by anyone who wants to run a model on their own computer. This page says
what works today, what doesn't, and how to tell us about a barrier.

## Where you can use it

- **The command line** (`run`, `chat`, `bench` and the other commands) is plain text in and out, with no
  colour-only information, so it works with screen readers and terminal magnifiers.
- **The API** (`serve`) is OpenAI-compatible, so you can use any client you already find accessible
  instead of the built-in chat page.
- **The chat page** is the part most likely to have barriers; see below.

## The chat page

What it does today:

- Buttons that only show an icon have text labels (`aria-label`), toggles report their state
  (`aria-pressed`, `aria-expanded`), and the decorative ghost is hidden from screen readers.
- It is a normal web page, so browser zoom and text size settings work, down to phone widths.
- Replies are rendered as text, not images; code blocks have a copy button.

Known limitations:

- **Streaming replies are not announced.** A screen reader doesn't read a reply as it arrives; you have
  to move to it once it's done.
- **Motion**: the ghost animates while the model works, and the page does not yet follow the system's
  "reduce motion" setting.
- **Keyboard focus** uses the browser's default outline, which is faint on some controls, and some menus
  have not been tested with the keyboard alone.
- There is no high-contrast or dark theme yet.
- It has not been tested with a screen reader end to end.

These are bugs, not decisions. If one of them is in your way, say so: that moves it up the list.

## Reporting a barrier

Open an issue with the [bug report form](https://github.com/affifuddin-hazam/ghost-infer/issues/new?template=bug_report.yml)
and put **Accessibility** in the title. Tell us what you were trying to do, what got in the way, and the
browser and assistive technology you use (for example NVDA with Firefox). You don't need a fix or a
technical description: "I can't reach the model picker with the keyboard" is a complete report.
