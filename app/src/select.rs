//! Text that can be selected with the mouse anywhere in the chat, as in a browser or Claude's apps: one selection
//! shared by every piece of text in the transcript (a reply's paragraphs and code, its thinking, tool output, the
//! user's words), dragged from one into the next and copied with Ctrl+C. iced's own text cannot be selected.
//!
//! Each piece is a [`Sel`], numbered in the order the view builds it ([`begin`] restarts the count), so the two
//! ends of the selection are (piece, line, byte) and a piece knows by its number alone whether it lies between
//! them. Each piece hit-tests the pointer itself (it gets the pointer in its own coordinates, inside nested
//! scrollables too). The [`Area`] around the transcript copies the pieces' text once, in order, and clears the
//! selection on a press no piece took. The hit testing and highlighting on cosmic-text follow iced_selection
//! (MIT, github.com/pml68/iced_selection), with one shared selection in place of one per widget.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use iced::advanced::graphics::text::cosmic_text;
use iced::advanced::graphics::text::Paragraph;
use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::{self as core_text, Paragraph as _, Renderer as _, Span};
use iced::advanced::widget::{self, tree, Tree, Widget};
use iced::advanced::{clipboard, mouse, overlay, Clipboard, Renderer as _, Shell};
use iced::{alignment, keyboard, Color, Element, Event, Font, Length, Pixels, Point, Rectangle, Size, Vector};

type Renderer = iced::Renderer;
type Theme = iced::Theme;

/// One end of the selection: a piece, a line of its text and a byte in that line.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct End {
    leaf: u32,
    line: usize,
    index: usize,
}

#[derive(Default)]
struct Global {
    anchor: Option<End>,
    focus: Option<End>,
    dragging: bool,
    claimed: bool,                 // a piece took this press
    copy: BTreeMap<u32, String>,   // the selected pieces' text, gathered on Ctrl+C
    shift: bool,
    color: Option<Color>,          // the selection's
}

thread_local! {
    static G: RefCell<Global> = RefCell::new(Global::default());
    static NEXT: Cell<u32> = const { Cell::new(0) };
}

/// Before the transcript is built: its pieces are numbered from 0 again, in the order they are made.
pub fn begin(color: Color) {
    NEXT.with(|n| n.set(0));
    G.with(|g| g.borrow_mut().color = Some(color));
}

/// Nothing selected (the chat changed under it, or Escape).
pub fn clear() {
    G.with(|g| {
        let mut g = g.borrow_mut();
        (g.anchor, g.focus, g.dragging) = (None, None, false);
    });
}

/// The selection's two ends in order, when it is not empty.
fn range() -> Option<(End, End)> {
    G.with(|g| {
        let g = g.borrow();
        match (g.anchor, g.focus) {
            (Some(a), Some(f)) if a != f => Some((a.min(f), a.max(f))),
            _ => None,
        }
    })
}

/// The part of piece `id` the selection covers: from (line, byte) to (line, byte), the end past the text when it
/// goes on into later pieces.
fn part(id: u32) -> Option<((usize, usize), (usize, usize))> {
    let (lo, hi) = range()?;
    if id < lo.leaf || id > hi.leaf {
        return None;
    }
    let s = if id == lo.leaf { (lo.line, lo.index) } else { (0, 0) };
    let e = if id == hi.leaf { (hi.line, hi.index) } else { (usize::MAX, usize::MAX) };
    Some((s, e))
}

fn floor(t: &str, mut i: usize) -> usize {
    i = i.min(t.len());
    while !t.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The text from `s` to `e` of a paragraph, its lines joined with newlines.
fn slice(p: &Paragraph, s: (usize, usize), e: (usize, usize)) -> String {
    let lines = &p.buffer().lines;
    let Some(last) = lines.len().checked_sub(1) else { return String::new() };
    let el = e.0.min(last);
    let mut out = String::new();
    for l in s.0..=el {
        let t = lines[l].text();
        let a = if l == s.0 { floor(t, s.1) } else { 0 };
        let b = if l == el { floor(t, e.1) } else { t.len() };
        if a < b {
            out.push_str(&t[a..b]);
        }
        if l < el {
            out.push('\n');
        }
    }
    out
}

/// Where in a paragraph drawn in `bounds` a point falls: (line, byte); above it its start, below it its end.
fn hit(p: &Paragraph, point: Point, bounds: Rectangle) -> (usize, usize) {
    let buffer = p.buffer();
    if point.y < bounds.y {
        return (0, 0);
    }
    if point.y > bounds.y + bounds.height {
        let l = buffer.lines.len().saturating_sub(1);
        return (l, buffer.lines.get(l).map_or(0, |x| x.text().len()));
    }
    let x = (point.x - bounds.x).max(0.0);
    let y = (point.y - bounds.y).min(bounds.height - 1.0).max(0.0);
    buffer.hit(x, y).map_or((0, 0), |c| (c.line, c.index))
}

/// The word around a byte of a line (a double click), or the run of spaces or the one mark there.
fn word(t: &str, i: usize) -> (usize, usize) {
    let i = floor(t, i);
    let kind = |c: char| if c.is_alphanumeric() || c == '_' { 0 } else if c.is_whitespace() { 1 } else { 2 };
    let Some(here) = t[i..].chars().next().or_else(|| t[..i].chars().next_back()) else { return (i, i) };
    let k = kind(here);
    if k == 2 {
        let start = if t[i..].chars().next().is_some() { i } else { i - here.len_utf8() };
        return (start, start + here.len_utf8());
    }
    let start = t[..i].char_indices().rev().take_while(|(_, c)| kind(*c) == k).last().map_or(i, |(j, _)| j);
    let end = t[i..].char_indices().find(|(_, c)| kind(*c) != k).map_or(t.len(), |(j, _)| i + j);
    (start, end)
}

/// A piece of selectable text, drawn as iced's rich text is (spans with their own fonts, colours, highlights
/// and links).
pub struct Sel<'a, Link: Clone + PartialEq + 'static, Message> {
    id: u32,
    spans: Box<dyn AsRef<[Span<'a, Link, Font>]> + 'a>,
    size: Option<Pixels>,
    line_height: core_text::LineHeight,
    font: Option<Font>,
    color: Option<Color>,
    width: Length,
    wrapping: core_text::Wrapping,
    on_link: Option<Box<dyn Fn(Link) -> Message + 'a>>,
}

/// Selectable rich text: the next piece of the transcript.
pub fn rich<'a, Link: Clone + PartialEq + 'static, Message>(spans: impl AsRef<[Span<'a, Link, Font>]> + 'a) -> Sel<'a, Link, Message> {
    let id = NEXT.with(|n| {
        let id = n.get();
        n.set(id + 1);
        id
    });
    Sel {
        id,
        spans: Box::new(spans),
        size: None,
        line_height: core_text::LineHeight::default(),
        font: None,
        color: None,
        width: Length::Shrink,
        wrapping: core_text::Wrapping::default(),
        on_link: None,
    }
}

/// Selectable plain text.
pub fn text<'a, Message>(t: impl core_text::IntoFragment<'a>) -> Sel<'a, (), Message> {
    rich(vec![Span::new(t)])
}

impl<'a, Link: Clone + PartialEq + 'static, Message> Sel<'a, Link, Message> {
    pub fn size(mut self, size: impl Into<Pixels>) -> Self {
        self.size = Some(size.into());
        self
    }
    pub fn line_height(mut self, h: impl Into<core_text::LineHeight>) -> Self {
        self.line_height = h.into();
        self
    }
    pub fn font(mut self, font: Font) -> Self {
        self.font = Some(font);
        self
    }
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
    pub fn width(mut self, width: impl Into<Length>) -> Self {
        self.width = width.into();
        self
    }
    pub fn wrapping(mut self, wrapping: core_text::Wrapping) -> Self {
        self.wrapping = wrapping;
        self
    }
    pub fn on_link_click(mut self, f: impl Fn(Link) -> Message + 'a) -> Self {
        self.on_link = Some(Box::new(f));
        self
    }
}

struct State<Link> {
    spans: Vec<Span<'static, Link, Font>>,
    paragraph: Paragraph,
    last_click: Option<mouse::Click>,
    hovered_link: Option<usize>,
    pressed_link: Option<usize>,
}

impl<'a, Link: Clone + PartialEq + 'static, Message> Widget<Message, Theme, Renderer> for Sel<'a, Link, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State<Link>>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::<Link> { spans: Vec::new(), paragraph: Paragraph::default(), last_click: None, hovered_link: None, pressed_link: None })
    }

    fn size(&self) -> Size<Length> {
        Size { width: self.width, height: Length::Shrink }
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) -> layout::Node {
        let state = tree.state.downcast_mut::<State<Link>>();
        let spans = self.spans.as_ref().as_ref();
        let (line_height, wrapping) = (self.line_height, self.wrapping);
        let size = self.size.unwrap_or_else(|| renderer.default_size());
        let font = self.font.unwrap_or_else(|| renderer.default_font());
        layout::sized(limits, self.width, Length::Shrink, |limits| {
            let bounds = limits.max();
            let text = |content| core_text::Text {
                content,
                bounds,
                size,
                line_height,
                font,
                align_x: core_text::Alignment::Default,
                align_y: alignment::Vertical::Top,
                shaping: core_text::Shaping::Advanced,
                wrapping,
            };
            if state.spans.as_slice() != spans {
                state.paragraph = Paragraph::with_spans(text(spans));
                state.spans = spans.iter().cloned().map(Span::to_static).collect();
            } else {
                let shape = core_text::Text { content: (), bounds, size, line_height, font, align_x: core_text::Alignment::Default, align_y: alignment::Vertical::Top, shaping: core_text::Shaping::Advanced, wrapping };
                match state.paragraph.compare(shape) {
                    core_text::Difference::None => {}
                    core_text::Difference::Bounds => state.paragraph.resize(bounds),
                    core_text::Difference::Shape => state.paragraph = Paragraph::with_spans(text(spans)),
                }
            }
            state.paragraph.min_bounds()
        })
    }

    fn draw(&self, tree: &Tree, renderer: &mut Renderer, _theme: &Theme, defaults: &renderer::Style, layout: Layout<'_>, _cursor: mouse::Cursor, viewport: &Rectangle) {
        let bounds = layout.bounds();
        if !bounds.intersects(viewport) {
            return;
        }
        let state = tree.state.downcast_ref::<State<Link>>();
        let translation = bounds.position() - Point::ORIGIN;
        let size = self.size.unwrap_or_else(|| renderer.default_size());
        let color = self.color.unwrap_or(defaults.text_color);
        // the spans' highlights (inline code) and underlines (links, the one under the pointer too)
        for (index, span) in self.spans.as_ref().as_ref().iter().enumerate() {
            let hovered = self.on_link.is_some() && Some(index) == state.hovered_link;
            if span.highlight.is_none() && !span.underline && !span.strikethrough && !hovered {
                continue;
            }
            let regions = state.paragraph.span_bounds(index);
            if let Some(h) = span.highlight {
                for r in &regions {
                    let r = Rectangle::new(r.position() - Vector::new(span.padding.left, span.padding.top), r.size() + Size::new(span.padding.x(), span.padding.y()));
                    renderer.fill_quad(Quad { bounds: r + translation, border: h.border, ..Default::default() }, h.background);
                }
            }
            if span.underline || span.strikethrough || hovered {
                let s = span.size.unwrap_or(size);
                let lh = span.line_height.unwrap_or(self.line_height).to_absolute(s);
                let c = span.color.unwrap_or(color);
                let baseline = translation + Vector::new(0.0, s.0 + (lh.0 - s.0) / 2.0);
                for r in &regions {
                    let y = if span.strikethrough { s.0 / 2.0 } else { s.0 * 0.08 };
                    renderer.fill_quad(Quad { bounds: Rectangle::new(r.position() + baseline - Vector::new(0.0, y), Size::new(r.width, 1.0)), ..Default::default() }, c);
                }
            }
        }
        // the selection, under the text
        if let Some((s, e)) = part(self.id) {
            let mark = G.with(|g| g.borrow().color).unwrap_or(Color { a: 0.3, ..Color::from_rgb(0.3, 0.5, 1.0) });
            for run in state.paragraph.buffer().layout_runs() {
                if run.line_i < s.0 || run.line_i > e.0 {
                    continue;
                }
                let a = cosmic_text::Cursor::new(run.line_i, if run.line_i == s.0 { s.1 } else { 0 });
                let b = cosmic_text::Cursor::new(run.line_i, if run.line_i == e.0 { e.1.min(run.text.len()) } else { run.text.len() });
                let (x, w) = run.highlight(a, b).unwrap_or((0.0, 0.0));
                // a whole empty line in the selection: a sliver, as a browser shows it
                let w = if w <= 0.0 && run.glyphs.is_empty() && run.line_i < e.0 { size.0 * 0.3 } else { w };
                if w > 0.0 {
                    let r = Rectangle { x: bounds.x + x, y: bounds.y + run.line_top, width: w, height: run.line_height };
                    renderer.fill_quad(Quad { bounds: r, ..Default::default() }, mark);
                }
            }
        }
        renderer.fill_paragraph(&state.paragraph, bounds.position(), color, *viewport);
    }

    fn update(&mut self, tree: &mut Tree, event: &Event, layout: Layout<'_>, cursor: mouse::Cursor, _renderer: &Renderer, _clipboard: &mut dyn Clipboard, shell: &mut Shell<'_, Message>, viewport: &Rectangle) {
        let state = tree.state.downcast_mut::<State<Link>>();
        let bounds = layout.bounds();
        let over = cursor.position_over(bounds).filter(|p| viewport.contains(*p));
        let before = state.hovered_link;
        state.hovered_link = match (over, &self.on_link) {
            (Some(p), Some(_)) => state.paragraph.hit_span(p - (bounds.position() - Point::ORIGIN)).filter(|&i| self.spans.as_ref().as_ref().get(i).is_some_and(|s| s.link.is_some())),
            _ => None,
        };
        if state.hovered_link != before {
            shell.request_redraw();
        }
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let Some(p) = over else { return };
                state.pressed_link = state.hovered_link;
                let click = mouse::Click::new(p, mouse::Button::Left, state.last_click);
                let (line, index) = hit(&state.paragraph, p, bounds);
                let lines = &state.paragraph.buffer().lines;
                G.with(|g| {
                    let mut g = g.borrow_mut();
                    let at = |line, index| Some(End { leaf: self.id, line, index });
                    match click.kind() {
                        mouse::click::Kind::Single => {
                            if !(g.shift && g.anchor.is_some()) {
                                g.anchor = at(line, index);
                            }
                            g.focus = at(line, index);
                            g.dragging = true;
                        }
                        mouse::click::Kind::Double => {
                            let (a, b) = word(lines.get(line).map_or("", |l| l.text()), index);
                            (g.anchor, g.focus, g.dragging) = (at(line, a), at(line, b), false);
                        }
                        mouse::click::Kind::Triple => {
                            let last = lines.len().saturating_sub(1);
                            (g.anchor, g.focus, g.dragging) = (at(0, 0), at(last, lines.get(last).map_or(0, |l| l.text().len())), false);
                        }
                    }
                    g.claimed = true;
                });
                state.last_click = Some(click);
                shell.capture_event();
                shell.request_redraw();
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if !G.with(|g| g.borrow().dragging) {
                    return;
                }
                // the pointer at this piece's height (and in sight): the selection's moving end in it
                let Some(p) = cursor.land().position() else { return };
                if p.y < bounds.y || p.y > bounds.y + bounds.height || p.y < viewport.y || p.y > viewport.y + viewport.height {
                    return;
                }
                let (line, index) = hit(&state.paragraph, p, bounds);
                let end = Some(End { leaf: self.id, line, index });
                if G.with(|g| std::mem::replace(&mut g.borrow_mut().focus, end) != end) {
                    shell.request_redraw();
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                // a click on a link, not a drag from it
                if let (Some(i), true) = (state.pressed_link.take(), range().is_none()) {
                    if Some(i) == state.hovered_link {
                        if let (Some(link), Some(f)) = (self.spans.as_ref().as_ref().get(i).and_then(|s| s.link.clone()), self.on_link.as_deref()) {
                            shell.publish(f(link));
                        }
                    }
                }
            }
            Event::Keyboard(keyboard::Event::KeyPressed { key: keyboard::Key::Character(c), modifiers, .. }) if c.as_str() == "c" && modifiers.command() => {
                if let Some((s, e)) = part(self.id) {
                    let t = slice(&state.paragraph, s, e);
                    G.with(|g| g.borrow_mut().copy.insert(self.id, t));
                }
            }
            _ => {}
        }
    }

    fn mouse_interaction(&self, tree: &Tree, layout: Layout<'_>, cursor: mouse::Cursor, _viewport: &Rectangle, _renderer: &Renderer) -> mouse::Interaction {
        let state = tree.state.downcast_ref::<State<Link>>();
        if state.hovered_link.is_some() {
            mouse::Interaction::Pointer
        } else if cursor.is_over(layout.bounds()) {
            mouse::Interaction::Text
        } else {
            mouse::Interaction::None
        }
    }
}

impl<'a, Link: Clone + PartialEq + 'static, Message: 'a> From<Sel<'a, Link, Message>> for Element<'a, Message> {
    fn from(s: Sel<'a, Link, Message>) -> Self {
        Element::new(s)
    }
}

/// Around the transcript: copies the selection on Ctrl+C (its pieces' text in order), and clears it on a press
/// no piece took (elsewhere in the chat, or in the message box) and on Escape.
pub struct Area<'a, Message> {
    content: Element<'a, Message>,
}

pub fn area<'a, Message>(content: impl Into<Element<'a, Message>>) -> Area<'a, Message> {
    Area { content: content.into() }
}

impl<'a, Message> Widget<Message, Theme, Renderer> for Area<'a, Message> {
    fn tag(&self) -> tree::Tag {
        self.content.as_widget().tag()
    }
    fn state(&self) -> tree::State {
        self.content.as_widget().state()
    }
    fn children(&self) -> Vec<Tree> {
        self.content.as_widget().children()
    }
    fn diff(&self, tree: &mut Tree) {
        self.content.as_widget().diff(tree);
    }
    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }
    fn size_hint(&self) -> Size<Length> {
        self.content.as_widget().size_hint()
    }
    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) -> layout::Node {
        self.content.as_widget_mut().layout(tree, renderer, limits)
    }
    fn operate(&mut self, tree: &mut Tree, layout: Layout<'_>, renderer: &Renderer, operation: &mut dyn widget::Operation) {
        self.content.as_widget_mut().operate(tree, layout, renderer, operation);
    }

    fn update(&mut self, tree: &mut Tree, event: &Event, layout: Layout<'_>, cursor: mouse::Cursor, renderer: &Renderer, clipboard: &mut dyn Clipboard, shell: &mut Shell<'_, Message>, viewport: &Rectangle) {
        let copy = matches!(event, Event::Keyboard(keyboard::Event::KeyPressed { key: keyboard::Key::Character(c), modifiers, .. }) if c.as_str() == "c" && modifiers.command());
        let press = matches!(event, Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)));
        G.with(|g| {
            let mut g = g.borrow_mut();
            if let Event::Keyboard(keyboard::Event::ModifiersChanged(m)) = event {
                g.shift = m.shift();
            }
            if press {
                g.claimed = false;
            }
            if copy {
                g.copy.clear();
            }
        });
        self.content.as_widget_mut().update(tree, event, layout, cursor, renderer, clipboard, shell, viewport);
        match event {
            _ if press && !G.with(|g| g.borrow().claimed) => {
                if range().is_some() {
                    shell.request_redraw();
                }
                clear();
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => G.with(|g| g.borrow_mut().dragging = false),
            Event::Keyboard(keyboard::Event::KeyPressed { key: keyboard::Key::Named(keyboard::key::Named::Escape), .. }) if range().is_some() => {
                clear();
                shell.request_redraw();
            }
            _ if copy => {
                let pieces = G.with(|g| std::mem::take(&mut g.borrow_mut().copy));
                if !pieces.is_empty() && !shell.is_event_captured() {
                    clipboard.write(clipboard::Kind::Standard, pieces.into_values().collect::<Vec<_>>().join("\n"));
                    shell.capture_event();
                }
            }
            _ => {}
        }
    }

    fn mouse_interaction(&self, tree: &Tree, layout: Layout<'_>, cursor: mouse::Cursor, viewport: &Rectangle, renderer: &Renderer) -> mouse::Interaction {
        self.content.as_widget().mouse_interaction(tree, layout, cursor, viewport, renderer)
    }

    fn draw(&self, tree: &Tree, renderer: &mut Renderer, theme: &Theme, style: &renderer::Style, layout: Layout<'_>, cursor: mouse::Cursor, viewport: &Rectangle) {
        self.content.as_widget().draw(tree, renderer, theme, style, layout, cursor, viewport);
    }

    fn overlay<'b>(&'b mut self, tree: &'b mut Tree, layout: Layout<'b>, renderer: &Renderer, viewport: &Rectangle, translation: Vector) -> Option<overlay::Element<'b, Message, Theme, Renderer>> {
        self.content.as_widget_mut().overlay(tree, layout, renderer, viewport, translation)
    }
}

impl<'a, Message: 'a> From<Area<'a, Message>> for Element<'a, Message> {
    fn from(a: Area<'a, Message>) -> Self {
        Element::new(a)
    }
}

#[cfg(test)]
mod tests {
    use super::word;

    #[test]
    fn a_double_click_takes_the_word() {
        assert_eq!(word("hello big world", 7), (6, 9));
        assert_eq!(word("hello big world", 15), (10, 15));
        assert_eq!(word("a, b", 1), (1, 2));
        assert_eq!(word("", 0), (0, 0));
    }
}
