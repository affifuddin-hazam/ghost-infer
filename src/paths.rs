//! Cross-platform data and cache directories for ghost-infer.
//!
//! Windows: %LOCALAPPDATA%\ghost-infer (both data and cache).
//! Linux:   $XDG_DATA_HOME/ghost-infer, else ~/.local/share/ghost-infer
//!          cache at $XDG_CACHE_HOME/ghost-infer, else ~/.cache/ghost-infer

use std::path::PathBuf;

/// The ghost-infer data directory: chats, settings, plugins, skills, memory,
/// LoRA adapters, downloaded models. `None` if no base directory is set.
pub fn data_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer"))
    }
    #[cfg(unix)]
    {
        if let Some(d) = std::env::var_os("XDG_DATA_HOME").filter(|s| !s.is_empty()) {
            return Some(PathBuf::from(d).join("ghost-infer"));
        }
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share").join("ghost-infer"))
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

/// The ghost-infer cache directory: compiled CUDA kernels, model-download
/// scratch, anything the program can regenerate. Windows reuses `data_dir` so
/// the installer's uninstall step sweeps everything with one key; XDG splits
/// cache and data by convention.
pub fn cache_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        data_dir()
    }
    #[cfg(unix)]
    {
        if let Some(d) = std::env::var_os("XDG_CACHE_HOME").filter(|s| !s.is_empty()) {
            return Some(PathBuf::from(d).join("ghost-infer"));
        }
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache").join("ghost-infer"))
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

/// The error message callers use when `data_dir()` returns `None`.
pub const NO_DATA_DIR: &str = if cfg!(windows) {
    "LOCALAPPDATA is not set"
} else {
    "neither XDG_DATA_HOME nor HOME is set"
};
