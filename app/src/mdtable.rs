//! A reply's Markdown split into text and GitHub-style tables, the tables drawn by the app (view::md_table).
//!
//! iced 0.14 draws a Markdown table inside a horizontal scrollable, which lays it out at an infinite width;
//! its row lines then become infinitely wide quads that tiny-skia refuses, and the app aborts ("Build quad
//! rectangle", issue #3). Its table rows keep their cells private, so a markdown::Viewer cannot draw them
//! another way: the tables are taken out of the text before iced parses it.

use iced::widget::markdown;
use iced::Alignment;

/// A piece of a reply.
#[derive(Debug)]
pub enum Part {
    Text(markdown::Content),
    Table(Table),
}

/// A table: per column its alignment, its header cell and its cells, each parsed as Markdown.
#[derive(Debug)]
pub struct Table {
    pub align: Vec<Alignment>,
    pub header: Vec<markdown::Content>,
    pub rows: Vec<Vec<markdown::Content>>,
}

/// Whether `md` holds a table (outside code blocks).
pub fn has_table(md: &str) -> bool {
    find(md).iter().any(|s| matches!(s, Seg::Table { .. }))
}

/// `md` as its text and its tables, in order.
pub fn parts(md: &str) -> Vec<Part> {
    find(md)
        .into_iter()
        .filter_map(|s| match s {
            Seg::Text(t) if t.trim().is_empty() => None,
            Seg::Text(t) => Some(Part::Text(markdown::Content::parse(&t))),
            Seg::Table { align, header, rows } => Some(Part::Table(Table {
                align,
                header: header.iter().map(|c| markdown::Content::parse(c)).collect(),
                rows: rows.iter().map(|r| r.iter().map(|c| markdown::Content::parse(c)).collect()).collect(),
            })),
        })
        .collect()
}

#[derive(Debug, PartialEq)]
enum Seg {
    Text(String),
    Table { align: Vec<Alignment>, header: Vec<String>, rows: Vec<Vec<String>> },
}

/// Splits `md` at its tables: a header line, a delimiter line (`|---|:--:|`) with as many cells, then the
/// lines with a `|` that follow. Lines in ``` or ~~~ code blocks are text.
fn find(md: &str) -> Vec<Seg> {
    let lines: Vec<&str> = md.split_inclusive('\n').collect();
    let mut segs = Vec::new();
    let mut text = String::new();
    let mut fence: Option<&str> = None;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim();
        if let Some(f) = fence {
            if t.starts_with(f) {
                fence = None;
            }
        } else if t.starts_with("```") || t.starts_with("~~~") {
            fence = Some(&t[..3]);
        } else if let Some(align) = lines.get(i + 1).and_then(|d| delimiter(d)).filter(|a| t.contains('|') && cells(t).len() == a.len()) {
            if !text.is_empty() {
                segs.push(Seg::Text(std::mem::take(&mut text)));
            }
            let n = align.len();
            let fit = |mut c: Vec<String>| {
                c.resize(n, String::new());
                c
            };
            let header = fit(cells(t));
            let mut rows = Vec::new();
            i += 2;
            while let Some(l) = lines.get(i).map(|l| l.trim()).filter(|l| !l.is_empty() && l.contains('|')) {
                rows.push(fit(cells(l)));
                i += 1;
            }
            segs.push(Seg::Table { align, header, rows });
            continue;
        }
        text.push_str(line);
        i += 1;
    }
    if !text.is_empty() {
        segs.push(Seg::Text(text));
    }
    segs
}

/// A delimiter line's column alignments (`:--` left, `:-:` centre, `--:` right), or None if it is not one.
fn delimiter(line: &str) -> Option<Vec<Alignment>> {
    let t = line.trim();
    if !t.contains('-') || !t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t')) {
        return None;
    }
    let c = cells(t);
    let ok = |c: &String| {
        let d = c.trim_start_matches(':').trim_end_matches(':');
        !d.is_empty() && d.chars().all(|ch| ch == '-')
    };
    if c.is_empty() || !c.iter().all(ok) || (c.len() == 1 && !t.contains('|')) {
        return None;
    }
    Some(c.iter().map(|c| match (c.starts_with(':'), c.ends_with(':')) {
        (true, true) => Alignment::Center,
        (false, true) => Alignment::End,
        _ => Alignment::Start,
    }).collect())
}

/// A table line's cells, trimmed: split at each `|` that is not escaped (`\|`) or in `code`; the outer pipes
/// are optional.
fn cells(line: &str) -> Vec<String> {
    let t = line.trim();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut code = false;
    let mut chars = t.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'|') => {
                cur.push('|');
                chars.next();
            }
            '`' => {
                code = !code;
                cur.push(c);
            }
            '|' if !code => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    if t.starts_with('|') {
        out.remove(0);
    }
    if t.ends_with('|') && !t.ends_with("\\|") && out.len() > 1 {
        out.pop();
    }
    out.into_iter().map(|c| c.trim().to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_a_table_out() {
        let md = "Fruits:\n\n| Name | Price |\n|:--|--:|\n| Apple | $1 |\n| Kiwi \\| gold | `a|b` |\n\nAfter.\n";
        let s = find(md);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Seg::Text("Fruits:\n\n".into()));
        assert_eq!(
            s[1],
            Seg::Table {
                align: vec![Alignment::Start, Alignment::End],
                header: vec!["Name".into(), "Price".into()],
                rows: vec![vec!["Apple".into(), "$1".into()], vec!["Kiwi | gold".into(), "`a|b`".into()]],
            }
        );
        assert_eq!(s[2], Seg::Text("\nAfter.\n".into()));
    }

    #[test]
    fn no_outer_pipes_and_short_rows() {
        let s = find("a | b | c\n--- | :-: | ---\n1 | 2\n");
        assert_eq!(s, vec![Seg::Table { align: vec![Alignment::Start, Alignment::Center, Alignment::Start], header: vec!["a".into(), "b".into(), "c".into()], rows: vec![vec!["1".into(), "2".into(), "".into()]] }]);
    }

    #[test]
    fn leaves_code_blocks_and_rules_alone() {
        let md = "```\n| a | b |\n|---|---|\n```\nText\n---\n";
        assert!(!has_table(md));
        assert_eq!(find(md), vec![Seg::Text(md.into())]);
        // a header line with a different number of cells than the delimiter: not a table
        assert!(!has_table("| a | b |\n|---|\n"));
    }

    #[test]
    fn a_table_being_written() {
        // the delimiter not there yet: text until it comes
        assert!(!has_table("| a | b |\n|--"));
        assert!(has_table("| a | b |\n|--|--|"));
    }
}
