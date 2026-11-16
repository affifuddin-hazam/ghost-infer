//! Access keys: what lets a browser or an app on another computer use this server.
//!
//! A server that only listens on this computer (127.0.0.1, the default) needs no key and works as it always has.
//! One that other devices can reach needs a key: given with --api-key (or GHOST_INFER_API_KEY), made in the
//! chat page's API & access settings, or made at the first start on a network address and printed as a sign-in
//! link. Once any key exists, every /v1 and /api request needs one, from this computer too: an SSH tunnel or a
//! Cloudflare tunnel arrives as 127.0.0.1, so where a request comes from proves nothing.
//!
//! How a request shows it may come in:
//!   Authorization: Bearer <key>      apps (every OpenAI client sends its API key this way); X-Api-Key works too
//!   Cookie: ghost_session=<id>       a browser that signed in once, with a key (POST /api/access/login, or the
//!                                    sign-in link /?key=<key>, which sets the cookie and leaves the key out of
//!                                    the address bar)
//!   X-Ghost-Token: <this run's token> programs of this computer's user: the token is in the run-token file
//!                                    beside the chats, which only that user can read
//!
//! Keys and sessions are kept in access.json beside the chats, as SHA-256 hashes: a key is shown once, when it
//! is made, and cannot be read back.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::json::{Json, num, obj, str};

/// How long a browser stays signed in without coming back.
const SESSION_DAYS: u64 = 30;

/// The cookie a signed-in browser carries.
pub const COOKIE: &str = "ghost_session";

#[derive(Clone)]
struct Key {
    id: String,
    name: String,
    hash: String,
    hint: String, // gi_…<last four>
    created: u64,
    last_used: u64,
}

#[derive(Clone)]
struct Session {
    id: String, // shown in the list; the cookie is another value, kept only as its hash
    hash: String,
    key: String, // the key it signed in with (revoking the key signs it out)
    agent: String,
    ip: String,
    created: u64,
    last_seen: u64,
}

#[derive(Default)]
struct Store {
    keys: Vec<Key>,
    sessions: Vec<Session>,
    network: bool, // listen on every address (0.0.0.0) when no --host is given
}

struct State {
    store: Store,
    cli: Option<String>, // the hash of --api-key / GHOST_INFER_API_KEY
    listening: String,   // host:port
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Who a request is.
#[derive(Debug, Clone, PartialEq)]
pub enum Who {
    Open,            // no key is needed here (a server on this computer only, with no keys)
    Local,           // this run's token: the desktop app, the monitor, other programs of this user
    Key(String),     // an app with a key (its id; "cli" for --api-key)
    Session(String), // a signed-in browser (its session id)
}

/// The credentials a request carried.
#[derive(Default, Debug, Clone)]
pub struct Creds {
    pub bearer: String,
    pub cookie: String, // the ghost_session cookie's value
    pub token: String,  // X-Ghost-Token
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn file() -> Option<PathBuf> {
    super::agent::data_dir().ok().map(|d| d.join("access.json"))
}

fn load() -> Store {
    let text = file().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
    let j = Json::parse(&text).unwrap_or(Json::Null);
    let s = |o: &Json, k: &str| o.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let n = |o: &Json, k: &str| o.get(k).and_then(Json::as_f64).unwrap_or(0.0) as u64;
    let list = |k: &str| j.get(k).and_then(Json::as_array).map(|a| a.to_vec()).unwrap_or_default();
    Store {
        keys: list("keys")
            .iter()
            .map(|k| Key { id: s(k, "id"), name: s(k, "name"), hash: s(k, "hash"), hint: s(k, "hint"), created: n(k, "created"), last_used: n(k, "last_used") })
            .filter(|k| !k.hash.is_empty())
            .collect(),
        sessions: list("sessions")
            .iter()
            .map(|x| Session { id: s(x, "id"), hash: s(x, "hash"), key: s(x, "key"), agent: s(x, "agent"), ip: s(x, "ip"), created: n(x, "created"), last_seen: n(x, "last_seen") })
            .filter(|x| !x.hash.is_empty() && now().saturating_sub(x.last_seen) < SESSION_DAYS * 86_400_000)
            .collect(),
        network: j.get("network").and_then(Json::as_bool).unwrap_or(false),
    }
}

fn save(s: &Store) {
    let keys = s.keys.iter().map(|k| obj(vec![("id", str(&k.id)), ("name", str(&k.name)), ("hash", str(&k.hash)), ("hint", str(&k.hint)), ("created", num(k.created as f64)), ("last_used", num(k.last_used as f64))])).collect();
    let sessions = s
        .sessions
        .iter()
        .map(|x| obj(vec![("id", str(&x.id)), ("hash", str(&x.hash)), ("key", str(&x.key)), ("agent", str(&x.agent)), ("ip", str(&x.ip)), ("created", num(x.created as f64)), ("last_seen", num(x.last_seen as f64))]))
        .collect();
    let j = obj(vec![("network", Json::Bool(s.network)), ("keys", Json::Arr(keys)), ("sessions", Json::Arr(sessions))]);
    if let Some(p) = file() {
        let tmp = p.with_extension("json.tmp");
        if std::fs::write(&tmp, j.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, &p);
        }
    }
}

/// Whether the saved settings ask for every address (used when no --host was given).
pub fn network_wanted() -> bool {
    load().network
}

fn is_loopback(host: &str) -> bool {
    let h = host.trim_start_matches('[').trim_end_matches(']');
    h == "localhost" || h.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Starts access control for a server on host:port. Returns a key made just now (to print as a sign-in link):
/// a server other devices can reach, with no key given and none saved, gets one.
pub fn init(host: &str, port: u16, cli: Option<String>) -> Option<String> {
    let mut store = load();
    let cli = cli.filter(|k| !k.trim().is_empty()).map(|k| sha256_hex(k.trim().as_bytes()));
    let mut made = None;
    if !is_loopback(host) && cli.is_none() && store.keys.is_empty() {
        let (key, k) = new_key("Made at first network start");
        store.keys.push(k);
        save(&store);
        made = Some(key);
    }
    *STATE.lock().unwrap() = Some(State { store, cli, listening: format!("{host}:{port}") });
    made
}

/// Whether requests need a key.
pub fn enabled() -> bool {
    STATE.lock().unwrap().as_ref().is_some_and(|s| s.cli.is_some() || !s.store.keys.is_empty())
}

/// Who the request is, or None when it needs a key and brought none that works.
pub fn check(c: &Creds, run_token: &str) -> Option<Who> {
    let local = !c.token.is_empty() && eq(&c.token, run_token);
    let mut guard = STATE.lock().unwrap();
    let Some(st) = guard.as_mut() else { return Some(if local { Who::Local } else { Who::Open }) };
    if st.cli.is_none() && st.store.keys.is_empty() {
        return Some(if local { Who::Local } else { Who::Open });
    }
    // a key or a signed-in browser first: a browser elsewhere also sends the page's token, and is still elsewhere
    if let Some(who) = keyed(st, c) {
        return Some(who);
    }
    local.then_some(Who::Local)
}

fn keyed(st: &mut State, c: &Creds) -> Option<Who> {
    let t = now();
    if !c.bearer.is_empty() {
        let h = sha256_hex(c.bearer.as_bytes());
        if st.cli.as_deref().is_some_and(|cli| eq(cli, &h)) {
            return Some(Who::Key("cli".into()));
        }
        if let Some(k) = st.store.keys.iter_mut().find(|k| eq(&k.hash, &h)) {
            let write = t.saturating_sub(k.last_used) > 60_000; // the file is written at most once a minute per key
            k.last_used = t;
            let id = k.id.clone();
            if write {
                save(&st.store);
            }
            return Some(Who::Key(id));
        }
    }
    if !c.cookie.is_empty() {
        let h = sha256_hex(c.cookie.as_bytes());
        if let Some(x) = st.store.sessions.iter_mut().find(|x| eq(&x.hash, &h)) {
            if t.saturating_sub(x.last_seen) < SESSION_DAYS * 86_400_000 {
                let write = t.saturating_sub(x.last_seen) > 60_000;
                x.last_seen = t;
                let id = x.id.clone();
                if write {
                    save(&st.store);
                }
                return Some(Who::Session(id));
            }
        }
    }
    None
}

/// Signs a browser in with a key: the new session's cookie value, or None for a wrong key.
pub fn login(key: &str, agent: &str, ip: &str) -> Option<String> {
    let h = sha256_hex(key.trim().as_bytes());
    let mut guard = STATE.lock().unwrap();
    let st = guard.as_mut()?;
    let key_id = if st.cli.as_deref().is_some_and(|cli| eq(cli, &h)) {
        "cli".to_string()
    } else {
        let k = st.store.keys.iter_mut().find(|k| eq(&k.hash, &h))?;
        k.last_used = now();
        k.id.clone()
    };
    let cookie = random_hex(32);
    let t = now();
    st.store.sessions.push(Session { id: random_hex(6), hash: sha256_hex(cookie.as_bytes()), key: key_id, agent: browser_name(agent), ip: ip.to_string(), created: t, last_seen: t });
    save(&st.store);
    Some(cookie)
}

/// Signs a session out.
pub fn logout(session: &str) -> bool {
    let mut guard = STATE.lock().unwrap();
    let Some(st) = guard.as_mut() else { return false };
    let n = st.store.sessions.len();
    st.store.sessions.retain(|x| x.id != session);
    let gone = st.store.sessions.len() < n;
    if gone {
        save(&st.store);
    }
    gone
}

fn new_key(name: &str) -> (String, Key) {
    let key = format!("gi_{}", random_b62(32));
    let k = Key { id: random_hex(6), name: name.to_string(), hash: sha256_hex(key.as_bytes()), hint: format!("gi_…{}", &key[key.len() - 4..]), created: now(), last_used: 0 };
    (key, k)
}

/// Makes a key: (the key, shown once; its id).
pub fn create_key(name: &str) -> Result<(String, String), String> {
    let name = name.trim();
    let name = if name.is_empty() { "Unnamed key" } else { name };
    let mut guard = STATE.lock().unwrap();
    let st = guard.as_mut().ok_or("the server has not started")?;
    let (key, k) = new_key(&name.chars().take(60).collect::<String>());
    let id = k.id.clone();
    st.store.keys.push(k);
    save(&st.store);
    Ok((key, id))
}

/// Revokes a key, and signs out every browser that came in with it.
pub fn revoke_key(id: &str) -> bool {
    let mut guard = STATE.lock().unwrap();
    let Some(st) = guard.as_mut() else { return false };
    let n = st.store.keys.len();
    st.store.keys.retain(|k| k.id != id);
    if st.store.keys.len() == n {
        return false;
    }
    st.store.sessions.retain(|x| x.key != id);
    save(&st.store);
    true
}

/// Listen on every address from the next start (or only on this computer). Turning it on makes a key when
/// there is none, so the server is never reachable without one; that key is returned to show once.
pub fn set_network(on: bool) -> Option<String> {
    let mut guard = STATE.lock().unwrap();
    let st = guard.as_mut()?;
    st.store.network = on;
    let mut made = None;
    if on && st.cli.is_none() && st.store.keys.is_empty() {
        let (key, k) = new_key("This browser");
        st.store.keys.push(k);
        made = Some(key);
    }
    save(&st.store);
    made
}

/// What the API & access settings show: {listening, network (saved setting), reachable (now), enabled,
/// cli_key, keys: [{id, name, hint, created, last_used}], sessions: [{id, agent, ip, created, last_seen, key,
/// this}], you: "local" | "open" | "key" | "session"}.
pub fn status(who: &Who) -> Json {
    let guard = STATE.lock().unwrap();
    let Some(st) = guard.as_ref() else { return Json::Null };
    let host = st.listening.rsplit_once(':').map_or(st.listening.as_str(), |(h, _)| h);
    let this = if let Who::Session(id) = who { id.as_str() } else { "" };
    let keys = st.store.keys.iter().map(|k| obj(vec![("id", str(&k.id)), ("name", str(&k.name)), ("hint", str(&k.hint)), ("created", num(k.created as f64)), ("last_used", num(k.last_used as f64))])).collect();
    let sessions = st
        .store
        .sessions
        .iter()
        .map(|x| {
            let key = if x.key == "cli" { "--api-key".to_string() } else { st.store.keys.iter().find(|k| k.id == x.key).map(|k| k.name.clone()).unwrap_or_default() };
            obj(vec![("id", str(&x.id)), ("agent", str(&x.agent)), ("ip", str(&x.ip)), ("created", num(x.created as f64)), ("last_seen", num(x.last_seen as f64)), ("key", str(&key)), ("this", Json::Bool(x.id == this))])
        })
        .collect();
    let you = match who {
        Who::Open => "open",
        Who::Local => "local",
        Who::Key(_) => "key",
        Who::Session(_) => "session",
    };
    obj(vec![
        ("listening", str(&st.listening)),
        ("reachable", Json::Bool(!is_loopback(host))),
        ("network", Json::Bool(st.store.network)),
        ("enabled", Json::Bool(st.cli.is_some() || !st.store.keys.is_empty())),
        ("cli_key", Json::Bool(st.cli.is_some())),
        ("tls", Json::Bool(false)),
        ("keys", Json::Arr(keys)),
        ("sessions", Json::Arr(sessions)),
        ("you", str(you)),
    ])
}

/// "Firefox on Linux" from a User-Agent header.
fn browser_name(ua: &str) -> String {
    let b = if ua.contains("Edg/") {
        "Edge"
    } else if ua.contains("Firefox/") {
        "Firefox"
    } else if ua.contains("Chrome/") {
        "Chrome"
    } else if ua.contains("Safari/") {
        "Safari"
    } else if ua.is_empty() {
        "A program"
    } else {
        "A browser"
    };
    let os = if ua.contains("iPhone") || ua.contains("iPad") {
        "iOS"
    } else if ua.contains("Android") {
        "Android"
    } else if ua.contains("Windows") {
        "Windows"
    } else if ua.contains("Mac OS") {
        "macOS"
    } else if ua.contains("Linux") {
        "Linux"
    } else {
        ""
    };
    if os.is_empty() { b.to_string() } else { format!("{b} on {os}") }
}

/// Compares two strings in time that does not depend on where they differ.
fn eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

// ---- randomness from the operating system

fn random_bytes(n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    #[cfg(windows)]
    {
        #[link(name = "advapi32")]
        unsafe extern "system" {
            #[link_name = "SystemFunction036"]
            fn RtlGenRandom(buf: *mut u8, len: u32) -> u8;
        }
        let ok = unsafe { RtlGenRandom(buf.as_mut_ptr(), n as u32) } != 0;
        assert!(ok, "the system gave no random bytes");
    }
    #[cfg(not(windows))]
    {
        use std::io::Read;
        std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut buf)).expect("the system gave no random bytes");
    }
    buf
}

pub fn random_hex(bytes: usize) -> String {
    random_bytes(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn random_b62(chars: usize) -> String {
    const A: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let mut out = String::new();
    while out.len() < chars {
        for b in random_bytes(chars) {
            if (b as usize) < 248 && out.len() < chars {
                out.push(A[b as usize % 62] as char); // 248 = 4 * 62: no bias
            }
        }
    }
    out
}

// ---- SHA-256 (FIPS 180-4)

pub fn sha256_hex(data: &[u8]) -> String {
    sha256(data).iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answers() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(sha256_hex(long), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }

    #[test]
    fn keys_look_right() {
        let (key, k) = new_key("test");
        assert!(key.starts_with("gi_") && key.len() == 35);
        assert_eq!(k.hash, sha256_hex(key.as_bytes()));
        assert!(k.hint.ends_with(&key[key.len() - 4..]));
        assert_ne!(random_hex(16), random_hex(16));
    }

    #[test]
    fn names_browsers() {
        assert_eq!(browser_name("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"), "Firefox on Linux");
        assert_eq!(browser_name("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"), "Safari on iOS");
        assert!(eq("abc", "abc") && !eq("abc", "abd") && !eq("abc", "ab"));
    }
}
