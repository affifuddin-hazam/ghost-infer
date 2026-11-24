//! Small cross-platform helpers used by the server and tools. Windows keeps
//! its exact previous behaviour (console-less child processes, `curl.exe`);
//! on Unix the same calls fall back to no-ops and the standard `curl` name.

use std::process::Command;

/// `curl.exe` on Windows (ships with Windows 10+), `curl` on other systems.
pub const CURL: &str = if cfg!(windows) { "curl.exe" } else { "curl" };

/// Keep a spawned child process from flashing a console window on Windows
/// (`CREATE_NO_WINDOW`). No-op on other platforms, where child processes do
/// not get their own console anyway.
pub trait NoConsole {
    fn no_console(&mut self) -> &mut Self;
}

impl NoConsole for Command {
    #[cfg(windows)]
    fn no_console(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        self.creation_flags(CREATE_NO_WINDOW)
    }
    #[cfg(not(windows))]
    fn no_console(&mut self) -> &mut Self {
        self
    }
}
