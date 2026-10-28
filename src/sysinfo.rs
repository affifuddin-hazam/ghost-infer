//! A snapshot of this computer for the chat page's get_system_info tool (GET /api/system): CPU, memory,
//! GPU, operating system and drives. Windows only for now (like the GPU backend); elsewhere it reports
//! what the standard library knows. The GPU is read through NVML (nvml.dll, installed with the NVIDIA
//! driver), which works from any thread, unlike CUDA's memory queries.

use crate::json::{Json, num, obj, str};

/// Everything as JSON, after `first` (the server's own part); parts that cannot be read are left out
/// rather than failing the whole report.
pub fn report(first: Vec<(&'static str, Json)>) -> Json {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let mut fields = first;
    fields.push(("cpu_threads", num(threads as f64)));
    #[cfg(windows)]
    win::fill(&mut fields);
    #[cfg(not(windows))]
    fields.push(("os", str(std::env::consts::OS)));
    obj(fields)
}

#[cfg(windows)]
mod win {
    use super::{Json, num, obj, str};
    use std::ffi::c_void;
    use std::sync::OnceLock;

    const GB: f64 = 1073741824.0;
    const HKEY_LOCAL_MACHINE: isize = 0x8000_0002u32 as i32 as isize;
    const RRF_RT_REG_SZ: u32 = 0x2;
    const RRF_RT_REG_DWORD: u32 = 0x10;

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_ext_virtual: u64,
    }

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_faults: u32,
        peak_working_set: usize,
        working_set: usize,
        rest: [usize; 6],
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        fn GetLogicalProcessorInformationEx(relation: i32, buffer: *mut u8, length: *mut u32) -> i32;
        fn GetLogicalDrives() -> u32;
        fn GetDriveTypeW(root: *const u16) -> u32;
        fn GetDiskFreeSpaceExW(dir: *const u16, free_to_caller: *mut u64, total: *mut u64, free: *mut u64) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegGetValueW(key: isize, sub: *const u16, value: *const u16, flags: u32, ty: *mut u32, data: *mut c_void, len: *mut u32) -> i32;
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }

    fn reg_string(sub: &str, value: &str) -> Option<String> {
        let mut buf = [0u16; 512];
        let mut len = (buf.len() * 2) as u32;
        let ok = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, wide(sub).as_ptr(), wide(value).as_ptr(), RRF_RT_REG_SZ, std::ptr::null_mut(), buf.as_mut_ptr() as *mut c_void, &mut len) };
        (ok == 0).then(|| String::from_utf16_lossy(&buf[..(len as usize / 2).saturating_sub(1)]).trim().to_string())
    }

    fn reg_dword(sub: &str, value: &str) -> Option<u32> {
        let (mut v, mut len) = (0u32, 4u32);
        let ok = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, wide(sub).as_ptr(), wide(value).as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(), &mut v as *mut u32 as *mut c_void, &mut len) };
        (ok == 0).then_some(v)
    }

    /// Physical cores: the processor-core records Windows reports (RelationProcessorCore = 0).
    fn physical_cores() -> Option<usize> {
        let mut len = 0u32;
        unsafe { GetLogicalProcessorInformationEx(0, std::ptr::null_mut(), &mut len) };
        let mut buf = vec![0u8; len as usize];
        if len == 0 || unsafe { GetLogicalProcessorInformationEx(0, buf.as_mut_ptr(), &mut len) } == 0 {
            return None;
        }
        let (mut at, mut cores) = (0usize, 0usize);
        while at + 8 <= len as usize {
            let size = u32::from_le_bytes(buf[at + 4..at + 8].try_into().unwrap()) as usize;
            if size == 0 {
                break;
            }
            cores += 1;
            at += size;
        }
        Some(cores)
    }

    fn gb(bytes: u64) -> Json {
        num((bytes as f64 / GB * 10.0).round() / 10.0)
    }

    pub fn fill(f: &mut Vec<(&'static str, Json)>) {
        const CPU: &str = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
        if let Some(name) = reg_string(CPU, "ProcessorNameString") {
            f.push(("cpu", str(&name)));
        }
        if let Some(cores) = physical_cores() {
            f.push(("cpu_cores", num(cores as f64)));
        }
        if let Some(mhz) = reg_dword(CPU, "~MHz") {
            f.push(("cpu_mhz", num(mhz as f64)));
        }

        let mut m = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, load: 0, total_phys: 0, avail_phys: 0, total_page: 0, avail_page: 0, total_virtual: 0, avail_virtual: 0, avail_ext_virtual: 0 };
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 {
            f.push(("ram_total_gb", gb(m.total_phys)));
            f.push(("ram_available_gb", gb(m.avail_phys)));
            f.push(("ram_in_use_percent", num(m.load as f64)));
        }
        let mut pm = ProcessMemoryCounters { cb: std::mem::size_of::<ProcessMemoryCounters>() as u32, page_faults: 0, peak_working_set: 0, working_set: 0, rest: [0; 6] };
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pm, pm.cb) } != 0 {
            f.push(("ghost_infer_ram_gb", gb(pm.working_set as u64)));
        }

        const NT: &str = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
        let build = reg_string(NT, "CurrentBuild").unwrap_or_default();
        // Windows 11 still calls itself "Windows 10" in ProductName; its builds start at 22000
        let mut product = reg_string(NT, "ProductName").unwrap_or_else(|| "Windows".into());
        if build.parse::<u32>().is_ok_and(|b| b >= 22000) {
            product = product.replace("Windows 10", "Windows 11");
        }
        let release = reg_string(NT, "DisplayVersion").or_else(|| reg_string(NT, "ReleaseId")).unwrap_or_default();
        f.push(("os", str(format!("{product} {release} (build {build})").trim())));

        if let Some(gpu) = nvml() {
            f.push(("gpus", Json::Arr(gpu)));
        }

        let mask = unsafe { GetLogicalDrives() };
        let drives: Vec<Json> = (0..26u8)
            .filter(|i| mask & (1 << i) != 0)
            .filter_map(|i| {
                let root = format!("{}:\\", (b'A' + i) as char);
                let w = wide(&root);
                if unsafe { GetDriveTypeW(w.as_ptr()) } != 3 {
                    return None; // fixed drives only (DRIVE_FIXED)
                }
                let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
                (unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut avail, &mut total, &mut free) } != 0)
                    .then(|| obj(vec![("drive", str(&root[..2])), ("total_gb", gb(total)), ("free_gb", gb(avail))]))
            })
            .collect();
        f.push(("drives", Json::Arr(drives)));
    }

    #[repr(C)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        used: u64,
    }
    #[repr(C)]
    struct NvmlUtilization {
        gpu: u32,
        memory: u32,
    }

    #[allow(clippy::type_complexity)]
    struct Nvml {
        count: unsafe extern "system" fn(*mut u32) -> i32,
        handle: unsafe extern "system" fn(u32, *mut *mut c_void) -> i32,
        name: unsafe extern "system" fn(*mut c_void, *mut u8, u32) -> i32,
        memory: unsafe extern "system" fn(*mut c_void, *mut NvmlMemory) -> i32,
        temperature: unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> i32,
        utilization: unsafe extern "system" fn(*mut c_void, *mut NvmlUtilization) -> i32,
        driver: unsafe extern "system" fn(*mut u8, u32) -> i32,
    }

    fn nvml_lib() -> Option<&'static Nvml> {
        static LIB: OnceLock<Option<Nvml>> = OnceLock::new();
        LIB.get_or_init(|| unsafe {
            let lib = LoadLibraryW(wide("nvml.dll").as_ptr());
            if lib.is_null() {
                return None;
            }
            let sym = |n: &str| {
                let p = GetProcAddress(lib, format!("{n}\0").as_ptr());
                (!p.is_null()).then_some(p)
            };
            let init: unsafe extern "system" fn() -> i32 = std::mem::transmute(sym("nvmlInit_v2")?);
            if init() != 0 {
                return None;
            }
            Some(Nvml {
                count: std::mem::transmute(sym("nvmlDeviceGetCount_v2")?),
                handle: std::mem::transmute(sym("nvmlDeviceGetHandleByIndex_v2")?),
                name: std::mem::transmute(sym("nvmlDeviceGetName")?),
                memory: std::mem::transmute(sym("nvmlDeviceGetMemoryInfo")?),
                temperature: std::mem::transmute(sym("nvmlDeviceGetTemperature")?),
                utilization: std::mem::transmute(sym("nvmlDeviceGetUtilizationRates")?),
                driver: std::mem::transmute(sym("nvmlSystemGetDriverVersion")?),
            })
        })
        .as_ref()
    }

    fn c_string(buf: &[u8]) -> String {
        String::from_utf8_lossy(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(buf.len())]).into_owned()
    }

    fn nvml() -> Option<Vec<Json>> {
        let n = nvml_lib()?;
        let mut driver = [0u8; 96];
        let driver = if unsafe { (n.driver)(driver.as_mut_ptr(), driver.len() as u32) } == 0 { c_string(&driver) } else { String::new() };
        let mut count = 0u32;
        if unsafe { (n.count)(&mut count) } != 0 {
            return None;
        }
        Some(
            (0..count)
                .filter_map(|i| {
                    let mut dev = std::ptr::null_mut();
                    if unsafe { (n.handle)(i, &mut dev) } != 0 {
                        return None;
                    }
                    let mut name = [0u8; 96];
                    unsafe { (n.name)(dev, name.as_mut_ptr(), name.len() as u32) };
                    let mut g = vec![("name", str(&c_string(&name))), ("driver", str(&driver))];
                    let mut mem = NvmlMemory { total: 0, free: 0, used: 0 };
                    if unsafe { (n.memory)(dev, &mut mem) } == 0 {
                        g.push(("vram_total_gb", gb(mem.total)));
                        g.push(("vram_used_gb", gb(mem.used)));
                        g.push(("vram_free_gb", gb(mem.free)));
                    }
                    let mut temp = 0u32;
                    if unsafe { (n.temperature)(dev, 0, &mut temp) } == 0 {
                        g.push(("temperature_c", num(temp as f64)));
                    }
                    let mut util = NvmlUtilization { gpu: 0, memory: 0 };
                    if unsafe { (n.utilization)(dev, &mut util) } == 0 {
                        g.push(("load_percent", num(util.gpu as f64)));
                    }
                    Some(obj(g))
                })
                .collect(),
        )
    }
}
