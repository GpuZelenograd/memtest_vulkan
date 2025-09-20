pub const POINTER_HEX_PRINT_WIDTH: usize = size_of::<usize>() * 2;

#[cfg(target_os = "linux")]
pub fn virtual_to_physical_for_self_process(
    #[expect(unused)] virt_addr: *const core::ffi::c_void,
) -> Option<usize> {
    None
}

#[cfg(not(target_os = "linux"))]
pub fn virtual_to_physical_for_self_process(
    #[expect(unused)] virt_addr: *const core::ffi::c_void,
) -> Option<usize> {
    None
}
