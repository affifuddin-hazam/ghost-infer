//! The preview pane's page: a web page the agent wrote (served by the server at /preview/...), shown beside
//! the chat in Edge's WebView2. The app draws everything else on the CPU; the page is a native view, a child
//! of the app's window, laid over the pane's area and moved there whenever the layout changes. It is made the
//! first time a page is shown and kept after (hidden while the pane is closed), so it costs nothing until then.
//!
//! The approach is iced_wry's (github.com/n1ght-hunter/iced_blocks): the view is built in window::run, on the
//! window's thread, and stays in a thread local, since a WebView cannot cross threads.

use std::cell::{Cell, RefCell};

use iced::Rectangle;

thread_local! {
    static VIEW: RefCell<Option<wry::WebView>> = const { RefCell::new(None) };
    /// The pane's area as last laid out (logical pixels), for a view made after it.
    static PLACE: Cell<Option<Rectangle>> = const { Cell::new(None) };
}

/// Whether the view has been made.
pub fn exists() -> bool {
    VIEW.with(|v| v.borrow().is_some())
}

/// Makes the view as a child of `window`, showing `url`, at the pane's place if it has been laid out.
pub fn create(window: &dyn iced::window::Window, url: &str) -> Result<(), String> {
    let handle = window.window_handle().map_err(|e| e.to_string())?;
    allow_child_drawing(&handle);
    let mut builder = wry::WebViewBuilder::new().with_url(url).with_focused(false).with_devtools(true);
    if let Some(r) = PLACE.with(Cell::get) {
        builder = builder.with_bounds(rect(r));
    }
    let view = builder.build_as_child(&handle).map_err(|e| e.to_string())?;
    VIEW.with(|v| *v.borrow_mut() = Some(view));
    Ok(())
}

/// Shows `url` (or the same page again: a reload).
pub fn navigate(url: &str) {
    with_view(|v| {
        let _ = v.load_url(url);
    });
}

pub fn set_visible(visible: bool) {
    with_view(|v| {
        let _ = v.set_visible(visible);
    });
}

/// Keyboard focus back to the app's own window (the page keeps it after a click in it otherwise).
pub fn focus_app() {
    with_view(|v| {
        let _ = v.focus_parent();
    });
}

/// Moves the view to the pane's area, when it changed (called as the pane is drawn).
pub fn place(bounds: Rectangle) {
    if PLACE.with(Cell::get) == Some(bounds) {
        return;
    }
    PLACE.with(|p| p.set(Some(bounds)));
    with_view(|v| {
        let _ = v.set_bounds(rect(bounds));
    });
}

fn with_view(f: impl FnOnce(&wry::WebView)) {
    VIEW.with(|v| {
        if let Some(view) = v.borrow().as_ref() {
            f(view);
        }
    });
}

fn rect(r: Rectangle) -> wry::Rect {
    wry::Rect {
        position: wry::dpi::LogicalPosition::new(r.x as f64, r.y as f64).into(),
        size: wry::dpi::LogicalSize::new(r.width.max(1.0) as f64, r.height.max(1.0) as f64).into(),
    }
}

/// The app's window clips its children out of what it paints (WS_CLIPCHILDREN); the view is drawn by its own
/// process, so that is taken off, as iced_wry does, or the window paints over the page.
fn allow_child_drawing(handle: &wry::raw_window_handle::WindowHandle<'_>) {
    let wry::raw_window_handle::RawWindowHandle::Win32(w) = handle.as_raw() else { return };
    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetWindowLongPtrW(hwnd: isize, index: i32) -> isize;
        fn SetWindowLongPtrW(hwnd: isize, index: i32, value: isize) -> isize;
    }
    const GWL_STYLE: i32 = -16;
    const WS_CLIPCHILDREN: isize = 0x0200_0000;
    let hwnd = w.hwnd.get();
    unsafe {
        let style = GetWindowLongPtrW(hwnd, GWL_STYLE);
        if style & WS_CLIPCHILDREN != 0 {
            SetWindowLongPtrW(hwnd, GWL_STYLE, style & !WS_CLIPCHILDREN);
        }
    }
}
