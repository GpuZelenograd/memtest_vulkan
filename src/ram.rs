pub const POINTER_HEX_PRINT_WIDTH: usize = size_of::<usize>() * 2;

pub fn budget() -> Option<usize> {
    None
}

#[cfg(target_os = "linux")]
pub fn virt_addr_details(
    #[expect(unused)] virt_addr: *const core::ffi::c_void,
    device_type: erupt::vk::PhysicalDeviceType,
) -> Option<String> {
    if device_type == erupt::vk::PhysicalDeviceType::DISCRETE_GPU {
        return None;
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub fn virt_addr_details(
    #[expect(unused)] virt_addr: *const core::ffi::c_void,
    #[expect(unused)] device_type: erupt::vk::PhysicalDeviceType,
) -> Option<String> {
    None
}
