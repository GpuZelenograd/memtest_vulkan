use std::sync::atomic::{AtomicBool, AtomicU8, Ordering::SeqCst};

static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
pub mod app_status {
    pub const SIGNATURE: u8 = 0b01000000u8;
    pub const SIGNATURE_MASK: u8 = 0b11100000u8;
    pub const INITED_OK: u8 = 0b00001u8; //init sequence complete
    pub const RUNTIME_ERRORS: u8 = 0b00010u8; //non-fatal errors were detected during runtime
    pub const RUNTIME_ABORT: u8 = 0b00100u8; //application exited in a non-expected way
    pub const USE_GRACEFUL_HANDLER: u8 = 0b01000u8; //application requested graceful exit handler
    pub const QUIT_JOB_REQUESTED: u8 = 0b10000u8; //External request to quit all application jobs
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
    if check_any_bits_set(fetch_status(), app_status::QUIT_JOB_REQUESTED) {
        return true;
    }
    return INTERRUPT_REQUESTED.swap(false, SeqCst);
}

pub fn immediate_exit(raise_abort_flag: bool) -> ! {
    if raise_abort_flag {
        raise_status_bit(app_status::RUNTIME_ABORT)
    }
    immediate_exit_with_status();
}

fn report_interrupt_request(quit_job: bool) {
    let graceful = check_any_bits_set(fetch_status(), app_status::USE_GRACEFUL_HANDLER);
    if quit_job {
        raise_status_bit(app_status::QUIT_JOB_REQUESTED);
    }
    if !graceful {
        //don't raise abort - in the absence of the graceful flag - interrupt request is the excpected exit method
        immediate_exit(false)
    }
    INTERRUPT_REQUESTED.swap(true, SeqCst);
}

pub fn setup_handler(graceful: bool) {
    if graceful {
        raise_status_bit(app_status::USE_GRACEFUL_HANDLER);
    }
    setup_handler_impl();
}

#[cfg(windows)]
pub fn setup_handler_impl() {
    use windows_sys::Win32::{self, System::Console};
    unsafe extern "system" fn os_handler(ctrltype: u32) -> Win32::Foundation::BOOL {
        let quit_job = ctrltype != Console::CTRL_C_EVENT;
        report_interrupt_request(quit_job);
        if quit_job {
            //don't return for events returning from which causes immediate termination
            std::thread::sleep(std::time::Duration::from_secs(25));
            immediate_exit(true);
        }
        1
    }
    unsafe {
        Console::SetConsoleCtrlHandler(Some(os_handler), 1);
    }
}

#[cfg(windows)]
fn immediate_exit_with_status() -> ! {
    unsafe { windows_sys::Win32::System::Threading::ExitProcess(APP_STATUS.load(SeqCst) as u32) }
}

#[cfg(unix)]
pub fn setup_handler_impl() {
    use nix::sys::signal;
    extern "C" fn os_handler(
        sig: nix::libc::c_int,
        _: *mut nix::libc::siginfo_t,
        _: *mut nix::libc::c_void,
    ) {
        let quit_job = sig != nix::libc::SIGINT;
        report_interrupt_request(quit_job);
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

#[cfg(unix)]
fn immediate_exit_with_status() -> ! {
    unsafe { nix::libc::_exit(APP_STATUS.load(SeqCst) as i32) }
}
