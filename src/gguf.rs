//! GGUF v2/v3 reader: header, metadata key/values and tensor table, over a memory map.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::I8(v) if v >= 0 => v as u64,
            Value::U16(v) => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::U32(v) => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::U64(v) => v,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(s) => {
                let one_line: String = s.chars().take(80).collect::<String>().replace('\n', "\\n");
                if s.chars().count() > 80 { write!(f, "\"{one_line}…\"") } else { write!(f, "\"{one_line}\"") }
            }
            Value::Array(a) => {
                let shown: Vec<String> = a.iter().take(4).map(|v| v.to_string()).collect();
                if a.len() > 4 { write!(f, "[{}, … ({} items)]", shown.join(", "), a.len()) } else { write!(f, "[{}]", shown.join(", ")) }
            }
            Value::Bool(b) => write!(f, "{b}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            other => write!(f, "{}", other.as_u64().map(|v| v.to_string()).unwrap_or_else(|| format!("{other:?}"))),
        }
    }
}

/// ggml tensor element types this engine knows about (ids from ggml.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    /// 256-value super-blocks: 16 sub-blocks of 16 with 6-bit scale, 3-bit values
    Q3K,
    /// 256-value super-blocks: 8 sub-blocks of 32 with 6-bit scale and min, 4-bit values
    Q4K,
    /// 256-value super-blocks: 8 sub-blocks of 32 with 6-bit scale and min, 5-bit values
    Q5K,
    /// 256-value super-blocks: 16 sub-blocks of 16 with 8-bit scale, 6-bit values
    Q6K,
    Other(u32),
}

impl GgmlType {
    fn from_id(id: u32) -> Self {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            other => GgmlType::Other(other),
        }
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> Option<(usize, usize)> {
        match self {
            GgmlType::F32 => Some((1, 4)),
            GgmlType::F16 => Some((1, 2)),
            GgmlType::Q8_0 => Some((32, 34)),
            GgmlType::Q3K => Some((256, 110)),
            GgmlType::Q4K => Some((256, 144)),
            GgmlType::Q5K => Some((256, 176)),
            GgmlType::Q6K => Some((256, 210)),
            GgmlType::Other(_) => None,
        }
    }

    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q3K => "Q3_K".into(),
            GgmlType::Q4K => "Q4_K".into(),
            GgmlType::Q5K => "Q5_K".into(),
            GgmlType::Q6K => "Q6_K".into(),
            GgmlType::Other(id) => format!("type#{id}"),
        }
    }

    /// K-quant formats (256-value super-blocks dotted against per-32 int8 activations).
    pub fn is_kquant(self) -> bool {
        matches!(self, GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K)
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset into the file.
    pub offset: usize,
    pub n_bytes: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub mmap: Mmap,
    /// where the file is (the GPU loader reads tensors straight from it)
    pub path: std::path::PathBuf,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len()).ok_or_else(|| format!("unexpected end of file at byte {}", self.pos))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16, String> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value, String> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let n = self.u64()? as usize;
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(elem_ty)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(format!("unknown metadata value type {other}")),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        // SAFETY: the model file is treated as read-only for the lifetime of the map.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("cannot map {}: {e}", path.display()))?;
        let mut r = Reader { buf: &mmap, pos: 0 };

        if r.u32()? != MAGIC {
            return Err(format!("{} is not a GGUF file", path.display()));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty)?;
            metadata.insert(key, value);
        }

        let mut raw = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ty = GgmlType::from_id(r.u32()?);
            let rel = r.u64()?;
            raw.push((name, dims, ty, rel));
        }

        let alignment = metadata.get("general.alignment").and_then(Value::as_u64).unwrap_or(DEFAULT_ALIGNMENT);
        let data_start = (r.pos as u64).div_ceil(alignment) * alignment;

        let mut tensors = Vec::with_capacity(n_tensors);
        for (name, dims, ty, rel) in raw {
            let n: u64 = dims.iter().product();
            let n_bytes = match ty.block() {
                Some((per_block, bytes)) => (n as usize / per_block) * bytes,
                None => 0, // unknown type: size unknown, only rejected if the model actually uses it
            };
            let offset = (data_start + rel) as usize;
            if ty.block().is_some() && offset + n_bytes > mmap.len() {
                return Err(format!("tensor {name} runs past the end of the file"));
            }
            tensors.push(TensorInfo { name, dims, ty, offset, n_bytes });
        }

        Ok(Gguf { version, metadata, tensors, mmap, path: path.to_path_buf() })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}
