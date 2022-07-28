use std::sync::atomic::{AtomicI32, AtomicU8, Ordering::SeqCst};

static CLOSE_REQUESTED: AtomicI32 = AtomicI32::new(0); //1 is resettable close request, 2 is non-resettable
pub mod app_status {
    pub const SIGNATURE: u8 = 0b01000000u8;
    pub const SIGNATURE_MASK: u8 = 0b11100000u8;
    pub const INITED_OK: u8 = 0b00001u8;
    pub const RUNTIME_ERRORS: u8 = 0b00010u8;
}

static APP_STATUS: AtomicU8 = AtomicU8::new(app_status::SIGNATURE); //1 is resettable close request, 2 is non-resettable

pub fn raise_status_bit(bit: u8) {
    APP_STATUS.fetch_or(bit, SeqCst);
}

pub fn fetch_status() -> u8 {
    return APP_STATUS.load(SeqCst) & !app_status::SIGNATURE_MASK;
}

pub fn check_any_bits_set(value: u8, bits: u8) -> bool {
    return (value & bits) != 0;
}

pub fn close_requested() -> bool {
    match CLOSE_REQUESTED.compare_exchange(1, 0, SeqCst, SeqCst) {
        Ok(value) => return value != 0,
        Err(value) => return value != 0,
    }
}

#[cfg(windows)]
pub fn setup_handler() {
    use windows_sys::Win32::{self, System::Console};
    unsafe extern "system" fn os_handler(ctrltype: u32) -> Win32::Foundation::BOOL {
        if ctrltype == Console::CTRL_C_EVENT {
            CLOSE_REQUESTED.store(1, SeqCst);
        } else {
            CLOSE_REQUESTED.store(2, SeqCst);
            //don't return for events returning from which causes immediate termination
            std::thread::sleep(std::time::Duration::from_secs(25));
        }
        1
    }
    unsafe {
        Console::SetConsoleCtrlHandler(Some(os_handler), 1);
    }
}

#[cfg(unix)]
pub fn setup_handler() {
    use nix::sys::signal;
    extern "C" fn os_handler(
        sig: nix::libc::c_int,
        _: *mut nix::libc::siginfo_t,
        _: *mut nix::libc::c_void,
    ) {
        if sig == nix::libc::SIGINT {
            CLOSE_REQUESTED.store(1, SeqCst);
        } else {
            CLOSE_REQUESTED.store(2, SeqCst);
        }
    }
    unsafe {
        let sig_action = signal::SigAction::new(
            signal::SigHandler::SigAction(os_handler),
            signal::SaFlags::empty(),
            signal::SigSet::empty(),
        );
        let _ = signal::sigaction(signal::SIGINT, &sig_action);
        let _ = signal::sigaction(signal::SIGTERM, &sig_action);
        let _ = signal::sigaction(signal::SIGHUP, &sig_action);
        let _ = signal::sigaction(signal::SIGQUIT, &sig_action);
    }
}
