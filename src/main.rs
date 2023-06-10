mod close;
mod input;
mod output;

use byte_strings::c_str;
use core::cmp::{max, min};
use erupt::{
    extensions::{ext_debug_utils, ext_memory_budget, ext_pci_bus_info},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use std::{
    env,
    ffi::{c_void, CStr, OsString},
    fmt,
    io::Write,
    mem, time,
};

struct CStrStaticPtr([*const std::os::raw::c_char; 1]);

impl core::ops::Deref for CStrStaticPtr {
    type Target = [*const std::os::raw::c_char; 1];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
unsafe impl Sync for CStrStaticPtr {}

const VK_LOADER_DEBUG: &str = "VK_LOADER_DEBUG";
const LAYER_KHRONOS_VALIDATION: &CStr = c_str!("VK_LAYER_KHRONOS_validation");
static LAYER_KHRONOS_VALIDATION_ARRAY: CStrStaticPtr =
    CStrStaticPtr([LAYER_KHRONOS_VALIDATION.as_ptr()]);
const GB: f32 = (1024 * 1024 * 1024) as f32;
const READ_SHADER: &[u32] = memtest_vulkan_build::compiled_vk_compute_spirv!(
    r#"
struct IOBuf
{
    err_bit1_idx: array<u32, 32>,
    err_bitcount: array<u32, 32>,
    mem_bitcount: array<u32, 32>,
    actual_ff: u32,
    actual_max: u32,
    actual_min: u32,
    idx_max: u32,
    idx_min: u32,
    done_iter_or_err: u32,
    iter: u32,
    calc_param: u32,
    first_elem: vec4<u32>
}

@group(0) @binding(0) var<storage, read_write> io: IOBuf;
@group(0) @binding(1) var<storage, read_write> test: array<vec4<u32>>;

fn addr_value_by_index(i:u32)->vec4<u32>
{
    let effective_index_of_u32 = i * 4u + io.calc_param;
    return vec4<u32>(effective_index_of_u32 + 1u, effective_index_of_u32 + 2u, effective_index_of_u32 + 3u, effective_index_of_u32 + 4u);
}

fn test_value_by_index(i:u32)->vec4<u32>
{
    let addrs : vec4<u32> = addr_value_by_index(i);
    let shifts : vec4<u32> = addrs % 31u;
    let rotated : vec4<u32> = (addrs << shifts) | (addrs >> (32u - shifts));
    return rotated;
}


let TEST_WINDOW_1D_MAX_GROUPS: u32 = 0x4000u;
let TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY: u32 = 0x2000u;//don't inner-multiply by window size

@compute @workgroup_size(64, 1, 1)
fn read(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    let addr_mod = effective_invocation_id % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let new_mod = (11 * effective_invocation_id + 999 * io.iter + io.calc_param +  7 * (effective_invocation_id / TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY)) % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let effective_addr = effective_invocation_id - addr_mod + new_mod; //make read order a bit rotated, not strictly sequential
    let actual_value : vec4<u32> = test[effective_addr];
    let expected_value : vec4<u32> = test_value_by_index(effective_addr);
    if any(actual_value != expected_value) {
        //slow path, executed only on errors found
        for(var i: i32 = 0; i < 4; i++) {
            let actual_u32 = actual_value[i];
            let error_mask = actual_u32 ^ expected_value[i];
            if error_mask == 0 {
                continue;
            }
            let one_bits = countOneBits(error_mask);
            if one_bits == 1
            {
                let bit_idx = firstLeadingBit(error_mask);
                atomicAdd(&io.err_bit1_idx[bit_idx], 1u);
            }
            atomicAdd(&io.err_bitcount[one_bits % 32u], 1u);
            let vec_addr: u32 = effective_addr * 4u + i;
            atomicMax(&io.idx_max, vec_addr);
            atomicMin(&io.idx_min, vec_addr);
            atomicMax(&io.done_iter_or_err, 0xFFFFFFFFu); //ERROR_STATUS
            let actual_bits = countOneBits(actual_u32);
            if actual_bits == 32
            {
                atomicAdd(&io.actual_ff, 1u);
            }
            else
            {
                atomicAdd(&io.mem_bitcount[actual_bits], 1u);
                atomicMax(&io.actual_max, actual_u32);
                atomicMin(&io.actual_min, actual_u32);
            }
        }
    }
    //assign done_iter_or_err only on specific index (performance reasons)
    if effective_addr == 0 {
        atomicMax(&io.done_iter_or_err, io.iter);
    } else if effective_addr == 1 {
        io.first_elem = expected_value;
    }
}

@compute @workgroup_size(64, 1, 1)
fn write(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    //make global_invocation_id processing specific memory addr different on writing compared to reading
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 8u * TEST_WINDOW_1D_MAX_GROUPS;//don't inner-multiply by window size
    let proccessed_mod = effective_invocation_id % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = effective_invocation_id + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
    test[proccessed_idx] = test_value_by_index(proccessed_idx);
}

@compute @workgroup_size(64, 1, 1)
fn emulate_write_bugs(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 8u * TEST_WINDOW_1D_MAX_GROUPS;//don't inner-multiply by window size
    let proccessed_mod = effective_invocation_id % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = effective_invocation_id + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
    test[proccessed_idx] = test_value_by_index(proccessed_idx);
    if proccessed_idx == 0xADBA {
        test[proccessed_idx][1] ^= 0x400000u;//error simulation for test
    }
}
"#
);

const WG_SIZE: i64 = 64;
const VEC_SIZE: usize = 4; //vector processed by single workgroup item
const ELEMENT_SIZE: i64 = std::mem::size_of::<u32>() as i64;
const ELEMENT_BIT_SIZE: usize = (ELEMENT_SIZE * 8) as usize;
const TEST_WINDOW_1D_MAX_GROUPS: i64 = 0x4000;
const TEST_WINDOW_SIZE_GRANULARITY: i64 =
    VEC_SIZE as i64 * WG_SIZE * ELEMENT_SIZE * TEST_WINDOW_1D_MAX_GROUPS * 8_i64;
const TEST_WINDOW_MAX_SIZE: i64 = 4 * 1024 * 1024 * 1024 - TEST_WINDOW_SIZE_GRANULARITY;
const TEST_DATA_KEEP_FREE: i64 = 400 * 1024 * 1024;
const MIN_WANTED_ALLOCATION: i64 = TEST_DATA_KEEP_FREE;
const ALLOCATION_TRY_STEP: i64 = TEST_DATA_KEEP_FREE;

struct ComputePipelines {
    read: vk::Pipeline,
    #[allow(dead_code)]
    write: vk::Pipeline,
    #[allow(dead_code)]
    emulate_write_bugs: vk::Pipeline,
}

#[derive(Default)]
struct U64HexDebug(i64);

impl fmt::Debug for U64HexDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

#[derive(Default)]
struct DriverVersionDebug(u32);

impl fmt::Debug for DriverVersionDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let upper10bits = self.0 >> 22;
        if upper10bits > 2 {
            // NVIDIA (0x75a04180 = 470.129.06) and intel-mesa-on-linux (0x5402006 = 21.2.6) driver versioning
            return write!(f, "v{}(0x{:X})", upper10bits, self.0);
        }
        let upper18bits = self.0 >> 14;
        if upper18bits > 2 && upper18bits < 500 {
            //intel-on-windows driver versioning (0x19453c = [30.0.]101.1340)
            return write!(f, "v{}(0x{:X})", upper18bits, self.0);
        }
        if self.0 < 64 {
            //basic small number versioning like llvm
            return write!(f, "ver{}", self.0);
        }
        // don't parse AMD versioning like "0x8000E6"
        return write!(f, "0x{:X}", self.0);
    }
}

#[derive(Copy, Clone)]
struct MostlyZeroArr<const LEN: usize>([u32; LEN]);

impl<const LEN: usize> fmt::Display for MostlyZeroArr<LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if LEN < 8 {
            for i in 0..LEN {
                write!(f, "0x{:08X} ", self.0[i])?;
            }
            return Ok(());
        }
        for i in 0..LEN {
            let next_row = i / 16;
            if next_row > 0
                && self.0[next_row * 16..next_row * 16 + 16]
                    .iter()
                    .all(|v| v == &0u32)
            {
                continue; //skip entire line of zeroes
            }
            if i % 16 == 0 && i != 0 {
                write!(f, "   0x{:X}? ", i / 16)?;
            }
            let vali = self.0[i];
            if vali > 999999u32 {
                write!(f, "{:3}m", vali / 1000000u32)?;
            } else if vali > 9999u32 {
                write!(f, "{:3}k", vali / 1000u32)?;
            } else if vali > 0 {
                write!(f, "{:4}", vali)?;
            } else {
                write!(f, "    ")?; //zero
            }
            if i % 16 == 15 {
                writeln!(f)?;
            } else if i % 4 == 3 {
                write!(f, "|")?;
            } else if i % 4 == 1 {
                write!(f, " ")?;
            }
        }
        Ok(())
    }
}

impl<const LEN: usize> std::default::Default for MostlyZeroArr<LEN> {
    fn default() -> Self {
        Self([0; LEN])
    }
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
struct IOBuf {
    err_bit1_idx: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    err_bitcount: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    mem_bitcount: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    actual_ff: u32,
    actual_max: u32,
    actual_min: u32,
    idx_max: u32,
    idx_min: u32,
    done_iter_or_err: u32,
    iter: u32,
    calc_param: u32,
    first_elem: MostlyZeroArr<VEC_SIZE>,
}

impl fmt::Display for IOBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "values range: 0x{:08X}..=0x{:08X}   FFFFFFFF-like count:{}    bit-level stats table:",
            self.actual_max, self.actual_min, self.actual_ff
        )?;
        writeln!(
            f,
            "         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF"
        )?;
        write!(f, "SinglIdx{}", self.err_bit1_idx)?;
        write!(f, "TogglCnt{}", self.err_bitcount)?;
        write!(f, "1sInValu{}", self.mem_bitcount)?;
        Ok(())
    }
}

impl IOBuf {
    fn for_initial_iteration() -> Self {
        let mut result = Self::default();
        result.prepare_next_iter_write();
        result
    }
    fn prepare_next_iter_write(&mut self) {
        self.reset_errors();
        self.iter += 1;
        self.set_calc_param_for_starting_window();
    }
    fn set_calc_param_for_starting_window(&mut self) {
        self.calc_param = self.iter.wrapping_mul(0x100107);
    }
    fn reset_errors(&mut self) {
        *self = IOBuf {
            iter: self.iter,
            calc_param: self.calc_param,
            idx_max: u32::MIN,
            idx_min: u32::MAX,
            actual_max: u32::MIN,
            actual_min: u32::MAX,
            ..Self::default()
        };
    }
    fn get_error_addresses_and_count(
        &self,
        buf_offset: i64,
    ) -> Option<(std::ops::RangeInclusive<U64HexDebug>, i64)> {
        if self.done_iter_or_err == self.iter {
            None
        } else {
            let total_errors = self.mem_bitcount.0.iter().sum::<u32>() as i64;
            Some((
                std::ops::RangeInclusive::<U64HexDebug>::new(
                    U64HexDebug(buf_offset + self.idx_min as i64 * ELEMENT_SIZE),
                    U64HexDebug(buf_offset + (self.idx_max + 1) as i64 * ELEMENT_SIZE - 1),
                ),
                total_errors,
            ))
        }
    }
    fn check_vec_first(&self) -> Result<(), Box<dyn std::error::Error>> {
        const TEST_IDX: u32 = 1;
        let addr: u32 = TEST_IDX * VEC_SIZE as u32 + self.calc_param + 1u32;
        let shift = addr % 31u32;
        let rotated = if shift > 0 {
            addr << shift | addr >> (32 - shift)
        } else {
            addr
        };
        if rotated != self.first_elem.0[0] {
            println!("{} 0x{:08X}", self, rotated);
            return Err("unexpected calculated value, maybe shader execution is broken".into());
        }
        Ok(())
    }
}

trait MapErrStr {
    type ValueType;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
    fn unwrap_or_display(self, env: &ProcessEnv) -> Self::ValueType
    where
        Self: Sized,
    {
        match self.err_as_str() {
            Err(e) => display_this_process_result(Some(e), env),
            Ok(v) => v,
        }
    }
}

trait MapErrRetryWithLowerMemory {
    type ValueType;
    fn err_retry_with_lower_memory(
        self,
        env: &ProcessEnv,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
}

impl<T> MapErrStr for std::result::Result<T, erupt::LoaderError> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|res| {
            let msg = match res {
                erupt::LoaderError::SymbolNotAvailable => {
                    "SymbolNotAvailable in Loader".to_string()
                }
                erupt::LoaderError::VulkanError(result) => format!("{}", result),
            } + " while getting "
                + std::any::type_name::<Self::ValueType>();
            msg.into()
        })
    }
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|res| {
            let msg = match res {
                erupt::LoaderError::SymbolNotAvailable => {
                    "SymbolNotAvailable in Loader".to_string()
                }
                erupt::LoaderError::VulkanError(result) => format!("{}", result),
            } + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            msg.into()
        })
    }
}

impl<T> MapErrStr for erupt::utils::VulkanResult<T> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg =
                res.to_string() + " while getting " + std::any::type_name::<Self::ValueType>();
            msg.into()
        })
    }
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg = res.to_string()
                + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            msg.into()
        })
    }
}
impl<T> MapErrRetryWithLowerMemory for erupt::utils::VulkanResult<T> {
    type ValueType = T;
    fn err_retry_with_lower_memory(
        self,
        env: &ProcessEnv,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg = res.to_string()
                + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            if !env.interactive
                && !close::check_any_bits_set(close::fetch_status(), close::app_status::INITED_OK)
            {
                if env.verbose {
                    println!("Retrying with lower memory due to {}", msg);
                }
                //immediate exit in non-interactive during init to initiate try with lower memory
                close::immediate_exit(true);
            }
            msg.into()
        })
    }
}

unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    eprintln!(
        "{}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );

    vk::FALSE
}
fn memory_requirements(
    device: &erupt::DeviceLoader,
    min_wanted_allocation: i64,
) -> Result<(vk::MemoryRequirements, vk::BufferCreateInfoBuilder), Box<dyn std::error::Error>> {
    let test_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(min_wanted_allocation as u64);
    let test_buffer =
        unsafe { device.create_buffer(&test_buffer_create_info, None) }.err_as_str()?;
    let test_mem_reqs = unsafe { device.get_buffer_memory_requirements(test_buffer) };
    unsafe { device.destroy_buffer(test_buffer, None) }; //buffer was used for getting memory reuirements. After allocation size may be smaller
    Ok((test_mem_reqs, test_buffer_create_info))
}

fn free_test_mem_and_buffers(
    device: &erupt::DeviceLoader,
    buffer: &mut Option<vk::Buffer>,
    memory: &mut Option<vk::DeviceMemory>,
) {
    if let Some(some_buffer) = buffer.take() {
        unsafe {
            device.destroy_buffer(some_buffer, None);
        }
    }
    if let Some(some_memory) = memory.take() {
        unsafe {
            device.free_memory(some_memory, None);
        }
    }
}
fn try_fill_default_mem_budget<Writer: std::io::Write>(
    loaded_devices: &LoadedDevices,
    env: &mut ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) {
    let selected_index = env.effective_index();
    let LoadedDevices(instance, _, _, devices_labeled_from_1) = &loaded_devices;

    if env.verbose {
        let _ = writeln!(
            log_dupler,
            "Loading memory info for selected device index {selected_index}...",
        );
    }
    if env.max_test_bytes > 0 || selected_index >= devices_labeled_from_1.len() {
        return;
    }

    let mut budget_structure: ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXT =
        Default::default();

    let mut memory_props = unsafe {
        instance.get_physical_device_memory_properties(
            devices_labeled_from_1[selected_index].physical_device,
        )
    };

    let mut budget_request = *vk::PhysicalDeviceMemoryProperties2Builder::new();

    if devices_labeled_from_1[selected_index].has_vk_1_1 {
        budget_request.p_next = &mut budget_structure
            as *mut ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXT
            as *mut c_void;
        let memory_props2 = unsafe {
            instance.get_physical_device_memory_properties2(
                devices_labeled_from_1[selected_index].physical_device,
                Some(budget_request),
            )
        };
        memory_props = memory_props2.memory_properties;
    }
    for i in 0..memory_props.memory_heap_count as usize {
        if env.verbose {
            let _ = writeln!(
                log_dupler,
                "heap size {:4.1}GB budget {:4.1}GB usage {:4.1}GB flags={:#?}",
                memory_props.memory_heaps[i].size as f32 / GB,
                budget_structure.heap_budget[i] as f32 / GB,
                budget_structure.heap_usage[i] as f32 / GB,
                memory_props.memory_heaps[i].flags,
            );
        }
        if !memory_props.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        {
            continue;
        }
        let mut heap_free = memory_props.memory_heaps[i].size as i64;
        let usage = budget_structure.heap_usage[i] as i64;
        if usage > 0 && usage < heap_free {
            heap_free -= usage;
        }
        let budget = budget_structure.heap_budget[i] as i64;
        if budget > 0 {
            heap_free = min(heap_free, budget);
        }
        env.max_test_bytes = max(env.max_test_bytes, heap_free - TEST_DATA_KEEP_FREE);
    }
}

fn prepare_and_test_device<Writer: std::io::Write>(
    instance: &erupt::InstanceLoader,
    selected: NamedComputeDevice,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    let queue_create_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(selected.queue_family_index)
        .queue_priorities(&[1.0])];

    let device_create_info =
        vk::DeviceCreateInfoBuilder::new().queue_create_infos(&queue_create_info);

    let memory_props =
        unsafe { instance.get_physical_device_memory_properties(selected.physical_device) };
    let device =
        match unsafe { DeviceLoader::new(instance, selected.physical_device, &device_create_info) }
        {
            Ok(device) => device,
            Err(e) => display_this_process_result(Some(e.into()), env),
        };
    let queue = unsafe { device.get_device_queue(selected.queue_family_index, 0) };

    let cmd_pool_info = vk::CommandPoolCreateInfoBuilder::new()
        .queue_family_index(selected.queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool =
        unsafe { device.create_command_pool(&cmd_pool_info, None) }.unwrap_or_display(env);

    let cmd_buf_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_info) }.unwrap_or_display(env);

    let desc_pool_sizes = &[vk::DescriptorPoolSizeBuilder::new()
        .descriptor_count(2)
        ._type(vk::DescriptorType::STORAGE_BUFFER)];
    let desc_pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(desc_pool_sizes)
        .max_sets(1);
    let desc_pool =
        unsafe { device.create_descriptor_pool(&desc_pool_info, None) }.unwrap_or_display(env);

    let desc_layout_bindings = &[
        vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let desc_layout_info =
        vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(desc_layout_bindings);
    let desc_layouts = [
        unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }
            .unwrap_or_display(env),
    ];

    let desc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(&desc_layouts);
    let desc_sets = unsafe { device.allocate_descriptor_sets(&desc_info) }.unwrap_or_display(env);

    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&desc_layouts);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .unwrap_or_display(env);

    let spv_code = Vec::from(READ_SHADER);
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv_code);
    let shader_mod =
        unsafe { device.create_shader_module(&create_info, None) }.unwrap_or_display(env);

    let pipeline_infos = [
        c_str!("read"),
        c_str!("write"),
        c_str!("emulate_write_bugs"),
    ]
    .map(|name| {
        let shader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(shader_mod)
            .name(name);
        vk::ComputePipelineCreateInfoBuilder::new()
            .layout(pipeline_layout)
            .stage(*shader_stage)
    });

    let pipelines =
        unsafe { device.create_compute_pipelines(Default::default(), &pipeline_infos, None) }
            .unwrap_or_display(env);
    let pipelines = ComputePipelines {
        read: pipelines[0],
        write: pipelines[1],
        emulate_write_bugs: pipelines[2],
    };

    if let Err(e) = test_device(
        &device,
        queue,
        cmd_bufs,
        desc_sets,
        &pipeline_layout,
        &pipelines,
        log_dupler,
        &selected.label,
        memory_props,
        env,
    ) {
        display_this_process_result(Some(e), env)
    }
    display_this_process_result(None, env)
}

fn test_device<Writer: std::io::Write>(
    device: &erupt::DeviceLoader,
    queue: vk::Queue,
    cmd_bufs: erupt::SmallVec<vk::CommandBuffer>,
    desc_sets: erupt::SmallVec<vk::DescriptorSet>,
    pipeline_layout: &vk::PipelineLayout,
    pipelines: &ComputePipelines,
    log_dupler: &mut output::LogDupler<Writer>,
    selected_label: &String,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    env: &ProcessEnv,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut allocation_size = env.max_test_bytes;
    if allocation_size < MIN_WANTED_ALLOCATION {
        return Err("requested test size is smaller than minimum wanted".into());
    }

    let io_data_size = mem::size_of::<IOBuf>() as vk::DeviceSize;

    let io_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(io_data_size);
    let io_buffer = unsafe { device.create_buffer(&io_buffer_create_info, None) }.err_as_str()?;
    let io_mem_reqs = unsafe { device.get_buffer_memory_requirements(io_buffer) };
    let mut io_mem_indices = Vec::new();
    for i in 0..memory_props.memory_type_count {
        //test buffer comptibility flags expressed as bitmask
        let suitable = (io_mem_reqs.memory_type_bits & (1 << i)) != 0;
        let memory_type = memory_props.memory_types[i as usize];
        if env.verbose && !memory_type.property_flags.is_empty() {
            let _ = writeln!(log_dupler, "{:2} {:?} ", i, memory_type);
        }
        if suitable
            && memory_type.property_flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        {
            io_mem_indices.push(i);
        }
    }

    // sorting by a flag value allows selection of index with the minimum count of new unknown flags
    let io_mem_index = io_mem_indices
        .into_iter()
        .min_by_key(|i| memory_props.memory_types[*i as usize].property_flags)
        .ok_or("This device lacks support for DEVICE_LOCAL+HOST_COHERENT memory type.")?;
    if env.verbose {
        let _ = writeln!(
            log_dupler,
            "CoherentIO memory          type {} inside heap {:?}",
            io_mem_index,
            memory_props.memory_heaps
                [memory_props.memory_types[io_mem_index as usize].heap_index as usize]
        );
    }

    let io_memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(io_mem_reqs.size)
        .memory_type_index(io_mem_index);
    let io_memory =
        unsafe { device.allocate_memory(&io_memory_allocate_info, None) }.err_as_str()?;

    let mapped: *mut IOBuf = unsafe {
        mem::transmute(
            device
                .map_memory(io_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::default())
                .err_as_str()?,
        )
    };
    unsafe { device.bind_buffer_memory(io_buffer, io_memory, 0) }
        .err_as_str_context("bind_buffer_memory")?;

    let (test_mem_reqs, test_buffer_create_info) =
        memory_requirements(device, MIN_WANTED_ALLOCATION)?;

    let test_mem_index = (0..memory_props.memory_type_count)
        .filter(|i| {
            //test buffer comptibility flags expressed as bitmask
            let suitable = (test_mem_reqs.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory_props.memory_types[*i as usize];
            suitable
                && memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .max_by_key(|mem_index| {
            let mem_type = memory_props.memory_types[*mem_index as usize];
            let heap_size = memory_props.memory_heaps[mem_type.heap_index as usize].size;
            // Among greatest heap_size select index with the minimum count of unknown flags
            (heap_size, std::cmp::Reverse(mem_type.property_flags))
        })
        .ok_or("DEVICE_LOCAL test memory type not available")?;

    unsafe {
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSetBuilder::new()
                .dst_set(desc_sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                    .buffer(io_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)])],
            &[],
        );
    }

    let fence =
        unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.err_as_str()?;

    let submit_info = &[vk::SubmitInfoBuilder::new().command_buffers(&cmd_bufs)];
    //all preparations except huge buffer allocation done. Now allocate huge buffer as a last step to minize chance of allocation failure for small structures

    let mut test_memory = None;
    let mut test_buffer = None;
    let mut test_window_count;
    let mut test_window_size;
    let mut warn_on_budget_alloc_fail = true;
    let mut execute_wait_queue;

    //The error state before all allocation tries
    let mut last_err: Box<dyn std::error::Error> =
        "No heap reports memory enough for testing".into();
    'memsize: loop {
        free_test_mem_and_buffers(device, &mut test_buffer, &mut test_memory);

        if allocation_size < MIN_WANTED_ALLOCATION {
            return Err(last_err);
        }

        let test_memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(allocation_size as u64)
            .memory_type_index(test_mem_index);
        if env.verbose {
            let _ = writeln!(
                log_dupler,
                "Trying {:7.3}GB buffer...",
                allocation_size as f32 / GB
            );
        }
        match unsafe { device.allocate_memory(&test_memory_allocate_info, None) }
            .err_retry_with_lower_memory(env, "allocate_memory")
        {
            Err(err) => last_err = err,
            Ok(some_memory) => {
                test_memory = Some(some_memory);
                let test_windows_max_size = TEST_WINDOW_SIZE_GRANULARITY;//TEST_WINDOW_MAX_SIZE;
                test_window_count = allocation_size / test_windows_max_size;
                if allocation_size % test_windows_max_size != 0 && (test_window_count + 1) * TEST_WINDOW_SIZE_GRANULARITY < allocation_size {
                    // increase window count so that nearly all allocation_size would be covered by a windows samller then test_windows_max_size;
                    // however dont reduce window to a size below TEST_WINDOW_SIZE_GRANULARITY
                    test_window_count += 1;
                }
                test_window_count = max(test_window_count, 2); //at least 2 windows: for testing rereads and rws
                test_window_size = allocation_size / test_window_count;
                test_window_size =
                    test_window_size - test_window_size % TEST_WINDOW_SIZE_GRANULARITY;
                let test_data_size = test_window_size * test_window_count;

                match unsafe {
                    device.create_buffer(&test_buffer_create_info.size(test_data_size as u64), None)
                }
                .err_retry_with_lower_memory(env, "create_buffer")
                {
                    Err(err) => last_err = err,
                    Ok(some_buffer) => {
                        test_buffer = Some(some_buffer);
                        match unsafe { device.bind_buffer_memory(some_buffer, some_memory, 0) }
                            .err_retry_with_lower_memory(env, "bind_buffer_memory")
                        {
                            Err(err) => last_err = err,
                            Ok(_) => {
                                execute_wait_queue = |buf_offset: i64, pipeline: vk::Pipeline| -> Result<(), Box<dyn std::error::Error>> {
                                    let test_element_count = (test_window_size / ELEMENT_SIZE) as u32;
                                    unsafe {
                                        device.update_descriptor_sets(
                                            &[vk::WriteDescriptorSetBuilder::new()
                                                .dst_set(desc_sets[0])
                                                .dst_binding(1)
                                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                                .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                                                    .buffer(test_buffer.unwrap())
                                                    .offset(buf_offset as u64)
                                                    .range(test_window_size as u64)])],
                                            &[],
                                        );
                                        let cmd_buf = cmd_bufs[0];
                                        device
                                            .begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default())
                                            .err_retry_with_lower_memory(env, "begin_command_buffer")?;
                                        device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
                                        device.cmd_bind_descriptor_sets(
                                            cmd_buf,
                                            vk::PipelineBindPoint::COMPUTE,
                                            *pipeline_layout,
                                            0,
                                            &desc_sets,
                                            &[],
                                        );
                                        device.cmd_dispatch(
                                            cmd_buf,
                                            (TEST_WINDOW_1D_MAX_GROUPS / WG_SIZE) as u32,
                                            test_element_count / VEC_SIZE as u32 / TEST_WINDOW_1D_MAX_GROUPS as u32,
                                            1,
                                        );
                                        device.end_command_buffer(cmd_buf).err_retry_with_lower_memory(env, "end_command_buffer")?;
                                        device
                                            .queue_submit(queue, submit_info, fence)
                                            .err_retry_with_lower_memory(env, "queue_submit")?;
                                        device
                                            .wait_for_fences(&[fence], true, u64::MAX)
                                            .err_retry_with_lower_memory(env, "wait_for_fences")?;
                                        device.reset_fences(&[fence]).err_retry_with_lower_memory(env, "reset_fences")?;
                                        Ok(())
                                    }
                                };
                                unsafe { std::ptr::write(mapped, IOBuf::for_initial_iteration()) }
                                //try to do initial memory fill to verify that allocation is really usable
                                let mut overall_exec_result = Ok(());
                                'window: for window_idx in 0..test_window_count {
                                    let test_offset = test_window_size * window_idx;
                                    if let Err(e) = execute_wait_queue(test_offset, pipelines.write)
                                    {
                                        overall_exec_result = Err(e);
                                        break 'window;
                                    }
                                    if let Err(e) = execute_wait_queue(test_offset, pipelines.read)
                                    {
                                        overall_exec_result = Err(e);
                                        break 'window;
                                    }
                                }
                                match overall_exec_result {
                                    Err(e) => last_err = e,
                                    Ok(()) => break 'memsize,
                                }
                            }
                        }
                    }
                }
            }
        }
        if env.verbose {
            let _ = writeln!(
                log_dupler,
                "Don't testing {:5.1}GB due to error: {}",
                allocation_size as f32 / GB,
                last_err
            );
        }
        if !env.interactive {
            close::immediate_exit(true);
        }
        if warn_on_budget_alloc_fail {
            warn_on_budget_alloc_fail = false;
            let _ = writeln!(log_dupler, "Failed allocating {:5.1}GB, trying to use smaller size. More system memory can help.", allocation_size as f32 / GB);
        }
        allocation_size -= ALLOCATION_TRY_STEP;
    }

    if env.verbose {
        let _ = writeln!(
            log_dupler,
            "Test memory size {:5.1}GB   type {:2}: {:?} {:?}",
            allocation_size as f32 / GB,
            test_mem_index,
            memory_props.memory_types[test_mem_index as usize],
            memory_props.memory_heaps
                [memory_props.memory_types[test_mem_index as usize].heap_index as usize]
        );
    }

    // allow write bugs emulation for testing purposes
    let emulate_write_bugs_iteration = env::var("MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or_default();
    let iter_count = 100000000; //by default exit after several days of testing
    let mut written_bytes = 0i64;
    let mut read_bytes = 0i64;
    let mut next_report_duration = time::Duration::from_secs(0);
    let extended_test_report_duration = time::Duration::from_secs(30);
    let mut reports_before_standard_done = 12i32;
    let mut write_duration = time::Duration::ZERO;
    let mut buffer_in = IOBuf::for_initial_iteration();
    let mut start = time::Instant::now();
    for iteration in 1..=iter_count {
        unsafe { std::ptr::write(mapped, buffer_in) }
        let write_start = time::Instant::now();
        for window_idx in 1..test_window_count {
            let test_offset = test_window_size * window_idx;
            unsafe {
                (*mapped).calc_param = buffer_in.calc_param + window_idx as u32 * 0x81_u32;
            }
            execute_wait_queue(
                test_offset,
                if iteration != emulate_write_bugs_iteration {
                    pipelines.write
                } else {
                    pipelines.emulate_write_bugs
                },
            )?;
        }
        written_bytes += test_window_size * (test_window_count - 1);
        write_duration += write_start.elapsed();
        let mut last_buffer_out: IOBuf;
        for window_idx in 0..test_window_count {
            let reread_mode_for_this_win = window_idx == 0;
            buffer_in.set_calc_param_for_starting_window();
            buffer_in.calc_param += window_idx as u32 * 0x81_u32;
            unsafe {
                std::ptr::write(
                    mapped,
                    if reread_mode_for_this_win {
                        IOBuf::for_initial_iteration()
                    } else {
                        buffer_in
                    },
                );
            }
            let test_offset = test_window_size * window_idx;
            execute_wait_queue(test_offset, pipelines.read)?;
            {
                unsafe {
                    last_buffer_out = std::ptr::read(mapped);
                }
                if let Some((error_range, total_errors)) =
                    last_buffer_out.get_error_addresses_and_count(test_offset)
                {
                    close::raise_status_bit(close::app_status::RUNTIME_ERRORS);
                    let test_elems = test_window_size / ELEMENT_SIZE;
                    write!(log_dupler,
                        "Error found. Mode {}, total errors 0x{:X} out of 0x{:X} ({:2.8}%)\nErrors address range: {:?}",
                        if reread_mode_for_this_win {
                            "NEXT_RE_READ"
                        } else {
                            "INITIAL_READ"
                        },
                        total_errors,
                        test_elems,
                        total_errors as f64/test_elems as f64 * 100.0f64,
                        error_range,
                    )?;
                    writeln!(
                        log_dupler,
                        "  iteration:{}\n{}",
                        last_buffer_out.iter, last_buffer_out
                    )?;
                }
                last_buffer_out.check_vec_first()?;
            }
        }
        read_bytes += test_window_size * test_window_count;
        let elapsed = start.elapsed();
        let stop_testing = close::close_requested();
        if elapsed > next_report_duration || stop_testing {
            let write_secs = write_duration.as_secs_f32();
            let passed_secs = elapsed.as_secs_f32() - write_secs;
            let write_speed_gbps = if write_secs > 0.0001 {
                written_bytes as f32 / GB / write_secs
            } else {
                0f32
            };
            let check_speed_gbps = if passed_secs > 0.0001 {
                read_bytes as f32 / GB / passed_secs
            } else {
                0f32
            };
            let second1 = time::Duration::from_secs(1);
            if next_report_duration.is_zero() {
                writeln!(log_dupler, "Standard 5-minute test of {}", selected_label)?;
                next_report_duration = second1; //2nd report after 1 second
            } else if next_report_duration == second1 {
                close::raise_status_bit(close::app_status::INITED_OK);
                next_report_duration = second1 * 5; //3rd report after 5 seconds
            } else {
                next_report_duration = extended_test_report_duration; //all later reports
            }
            if reports_before_standard_done == 0 {
                let has_errors = close::check_any_bits_set(
                    close::fetch_status(),
                    close::app_status::RUNTIME_ERRORS,
                );
                match has_errors {
                    true => writeln!(log_dupler, "Standard 5-minute test fail - ERRORS FOUND"),
                    false => writeln!(log_dupler, "Standard 5-minute test PASSed! Just press Ctrl+C unless you plan long test run."),
                }?;
                writeln!(
                    log_dupler,
                    "Extended endless test started; testing more than 2 hours is usually unneeded"
                )?;
                writeln!(
                    log_dupler,
                    "use Ctrl+C to stop it when you decide it's enough"
                )?;
            } else {
                writeln!(log_dupler, "{:7} iteration. Passed {:7.4} seconds  written:{:7.1}GB{:6.1}GB/sec        checked:{:7.1}GB{:6.1}GB/sec", iteration, elapsed.as_secs_f32(), written_bytes as f32 / GB, write_speed_gbps, read_bytes as f32 / GB, check_speed_gbps)?;
            }
            reports_before_standard_done -= 1;
            if reports_before_standard_done == 0 {
                // The last iteration before report has a sleep before it to test hot gpu behaviour
                // in a situation of load pause and low-performance memory frequency
                std::thread::sleep(next_report_duration / 2);
            }
            written_bytes = 0i64;
            read_bytes = 0i64;
            write_duration = time::Duration::ZERO;
            start = time::Instant::now();
        }
        if stop_testing {
            let _ = writeln!(log_dupler, "received user interruption, testing stopped");
            break;
        }
        buffer_in.prepare_next_iter_write();
    }
    // Cleanup & Destruction
    unsafe {
        device.device_wait_idle().err_as_str()?;

        free_test_mem_and_buffers(device, &mut test_buffer, &mut test_memory);

        device.destroy_buffer(io_buffer, None);
        device.unmap_memory(io_memory);
        device.free_memory(io_memory, None);

        device.destroy_fence(fence, None);
    }
    close::declare_exit_due_timeout();
    Ok(())
}

struct NamedComputeDevice {
    label: String,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    has_vk_1_1: bool,
}

fn load_instance<Writer: std::io::Write>(
    verbose: bool,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<
    (
        erupt::InstanceLoader,
        erupt::EntryLoader,
        vk::DebugUtilsMessengerEXT,
    ),
    Box<dyn std::error::Error>,
> {
    let override_vk_loader_debug = env::var_os(VK_LOADER_DEBUG).is_none();
    if verbose && override_vk_loader_debug {
        env::set_var(VK_LOADER_DEBUG, "error,warn");
    }

    let mut entry = EntryLoader::new()?;
    if verbose {
        let _ = writeln!(
            log_dupler,
            "Verbose feature enabled (or 'verbose' found in name). Vulkan instance {}.{}.{}",
            vk::api_version_major(entry.instance_version()),
            vk::api_version_minor(entry.instance_version()),
            vk::api_version_patch(entry.instance_version())
        );
        for (idx, prop) in unsafe { entry.enumerate_instance_layer_properties(None) }
            .err_as_str()
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            if idx == 0 {
                let _ = writeln!(log_dupler, "Available: ");
            } else {
                let _ = write!(log_dupler, ", ");
            }

            let _ = write!(log_dupler, "{}", unsafe {
                CStr::from_ptr(prop.layer_name.as_ptr())
                    .to_str()
                    .unwrap_or("Invalid property_name")
            });
        }
        let _ = writeln!(log_dupler);
        for (idx, ext) in unsafe { entry.enumerate_instance_extension_properties(None, None) }
            .err_as_str()
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            if idx == 0 {
                let _ = write!(log_dupler, "Extensions: ");
            } else {
                let _ = write!(log_dupler, ", ");
            }

            let _ = write!(log_dupler, "{}", unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr())
                    .to_str()
                    .unwrap_or("Invalid extension_name")
            });
        }
        let _ = writeln!(log_dupler);
        let _ = writeln!(log_dupler);
    }

    let instance_extensions = vec![ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME];

    let app_info = vk::ApplicationInfoBuilder::new().api_version(vk::API_VERSION_1_1);
    let instance_create_info = vk::InstanceCreateInfoBuilder::new()
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&*LAYER_KHRONOS_VALIDATION_ARRAY)
        .application_info(&app_info);

    let mut messenger = vk::DebugUtilsMessengerEXT::null();
    match unsafe {
        InstanceLoader::new(&entry, &instance_create_info)
            .err_as_str_context("instance with validation")
    } {
        Ok(instance_with_validation) => {
            let mut severity = ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                | ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT;
            if verbose {
                severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::INFO_EXT;
                //lists all extensions, very verbose
                //severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT;
            }
            let create_info = ext_debug_utils::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                .message_severity(severity)
                .message_type(ext_debug_utils::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(debug_callback));
            messenger = unsafe {
                instance_with_validation.create_debug_utils_messenger_ext(&create_info, None)
            }
            .result()
            .unwrap_or_default();
            return Ok((instance_with_validation, entry, messenger));
        }
        Err(e) => {
            if verbose {
                let _ = writeln!(log_dupler, "Not using validation layers due to {e}");
            }
        }
    }

    //fallback creation without validation and extensions
    let simple_instance_try = unsafe {
        InstanceLoader::new(
            &entry,
            &vk::InstanceCreateInfoBuilder::new().application_info(&app_info),
        )
    }
    .err_as_str_context("instance. Try specifying icd.json via VK_DRIVER_FILES env var");
    match simple_instance_try {
        Ok(instance) => Ok((instance, entry, messenger)),
        Err(e) => {
            if !override_vk_loader_debug {
                return Err(e);
            }
            drop(entry);
            //retry instance creation with loader debuf enabled
            env::set_var(VK_LOADER_DEBUG, "all");
            entry = EntryLoader::new()?;
            let debug_instance_try = unsafe {
                InstanceLoader::new(
                    &entry,
                    &vk::InstanceCreateInfoBuilder::new().application_info(&app_info),
                )
            }
            .map_err(|_second_error_ignired| e)?;
            Ok((debug_instance_try, entry, messenger))
        }
    }
}
//InstanceLoader must be dropped after EntryLoader
struct LoadedDevices(
    erupt::InstanceLoader,
    erupt::EntryLoader,
    vk::DebugUtilsMessengerEXT,
    Vec<NamedComputeDevice>,
);

impl Drop for LoadedDevices {
    fn drop(&mut self) {
        let LoadedDevices(instance, _, messenger, _) = self;
        unsafe {
            println!("Destroying vk instance...");
            if !messenger.is_null() {
                instance.destroy_debug_utils_messenger_ext(*messenger, None);
            }
            instance.destroy_instance(None);
        }
    }
}
fn list_devices_ordered_labaled_from_1<Writer: std::io::Write>(
    verbose: bool,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<LoadedDevices, Box<dyn std::error::Error>> {
    let (instance, entry, messenger) = load_instance(verbose, log_dupler)?;
    let mut compute_capable_devices: Vec<_> = unsafe { instance.enumerate_physical_devices(None) }
        .err_as_str()?
        .into_iter()
        .filter_map(|physical_device| unsafe {
            let queue_family = match instance
                .get_physical_device_queue_family_properties(physical_device, None)
                .into_iter()
                .position(|properties| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                Some(queue_family) => queue_family as u32,
                None => return None,
            };

            let mut pci_props_structure: ext_pci_bus_info::PhysicalDevicePCIBusInfoPropertiesEXT =
                Default::default();
            let mut properties = instance.get_physical_device_properties(physical_device);
            let effective_version = (
                vk::api_version_major(properties.api_version),
                vk::api_version_minor(properties.api_version),
            );

            //older vulkan implementations like broadcom on RaspberryPi lacks vk_1_1 support even if application requested it
            let has_vk_1_1 = effective_version >= (1, 1);
            if has_vk_1_1 {
                let mut pci_structure_request = *vk::PhysicalDeviceProperties2Builder::new();
                pci_structure_request.p_next = &mut pci_props_structure
                    as *mut ext_pci_bus_info::PhysicalDevicePCIBusInfoPropertiesEXT
                    as *mut c_void;

                properties = instance
                    .get_physical_device_properties2(physical_device, Some(pci_structure_request))
                    .properties;
            }
            let memory_props = instance.get_physical_device_memory_properties(physical_device);

            let mut max_local_heap_size = 0i64;
            for i in 0..memory_props.memory_heap_count as usize {
                if !memory_props.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                {
                    continue;
                }
                max_local_heap_size = max(
                    max_local_heap_size,
                    memory_props.memory_heaps[i].size as i64,
                );
            }

            Some((
                physical_device,
                queue_family,
                properties,
                max_local_heap_size,
                pci_props_structure,
                has_vk_1_1,
            ))
        })
        .collect();
    compute_capable_devices.sort_by_key(|(_, _, props, _, pci_props, _)| {
        let negative_bus_for_reverse_ordering = -(pci_props.pci_bus as i32);
        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => (0, negative_bus_for_reverse_ordering),
            vk::PhysicalDeviceType::INTEGRATED_GPU => (1, negative_bus_for_reverse_ordering),
            _ => (2, negative_bus_for_reverse_ordering),
        }
    });
    let mut numbered_devices: Vec<NamedComputeDevice> = Vec::new();
    for (i, d) in compute_capable_devices.iter().enumerate() {
        let props = d.2;
        let pci_props = d.4;
        let api_info = if verbose {
            std::format!(
                "API {}.{}.{}  {:?}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
                DriverVersionDebug(props.driver_version),
            )
        } else {
            String::new()
        };
        numbered_devices.push(NamedComputeDevice {
            label: std::format!(
                "{}: Bus=0x{:02X}:{:02X} DevId=0x{:04X} {api_info}  {}GB {}",
                i + 1,
                pci_props.pci_bus,
                pci_props.pci_device,
                props.device_id,
                (d.3 as f32 / GB).ceil(),
                unsafe {
                    CStr::from_ptr(props.device_name.as_ptr())
                        .to_str()
                        .unwrap_or("Invalid device_name")
                }
            ),
            physical_device: d.0,
            queue_family_index: d.1,
            has_vk_1_1: d.5,
        });
    }
    Ok(LoadedDevices(instance, entry, messenger, numbered_devices))
}

fn prompt_for_label(verbose: bool) -> Option<usize> {
    let mut device_test_index = Some(0usize);
    let prompt_start = time::Instant::now();
    let mut prompt_duration: Option<time::Duration> = Some(time::Duration::from_secs(10));

    let mut input_reader = input::Reader::default();
    let no_timer_prompt =
        String::from("                                                   Override index to test:");
    loop {
        let mut prompt = &no_timer_prompt;
        let formatted_prompt: String;
        if let Some(effective_duration) = prompt_duration {
            if effective_duration < prompt_start.elapsed() {
                println!();
                println!("    ...first device autoselected");
                break;
            } else {
                let duration_left = effective_duration - prompt_start.elapsed();
                formatted_prompt = std::format!(
                    "(first device will be autoselected in {} seconds)   Override index to test:",
                    duration_left.as_secs()
                );
                prompt = &formatted_prompt;
            }
        }
        match input_reader.input_digit_step(prompt, &time::Duration::from_millis(250)) {
            Ok(input::ReaderEvent::Edited) => prompt_duration = None,
            Ok(input::ReaderEvent::Canceled) => {
                println!();
                device_test_index = None;
                break;
            }
            Ok(input::ReaderEvent::Completed) => {
                match input_reader.current_input.len()
                    == input_reader.current_input.matches('0').count()
                {
                    true => {
                        //empty or all zeroes
                        println!();
                        println!("    ...testing default device confirmed");
                    }
                    false => match input_reader.current_input.parse::<usize>() {
                        Ok(parsed_idx) => {
                            device_test_index = Some(parsed_idx);
                            println!();
                        }
                        Err(_) => {
                            input_reader.current_input.clear();
                            continue;
                        }
                    },
                }
                break;
            }
            Ok(input::ReaderEvent::Timeout) => {} //just redraw prompt
            Err(e) => {
                //if input machinery doesnt work treat it as nothing was enetred
                if verbose {
                    println!("Input machinery failure: {e}");
                }
                break;
            }
        }
    }
    drop(input_reader);
    device_test_index
}
struct TestStatus {
    test_status: u8,
}
fn test_in_this_process<Writer: std::io::Write>(
    mut loaded_devices: LoadedDevices,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    let LoadedDevices(instance, _, _, devices_labeled_from_1) = &mut loaded_devices;
    let selected_index = env.effective_index();
    if selected_index >= devices_labeled_from_1.len() {
        display_this_process_result(Some("No device at given index".into()), env)
    }

    if env.max_test_bytes == 0 {
        display_this_process_result(Some("Failed determining memory budget".into()), env)
    }

    prepare_and_test_device(
        instance,
        devices_labeled_from_1.swap_remove(selected_index),
        env,
        log_dupler,
    )
}

enum SubprocessMode {
    NotExeced,
    FailedRetryLowerMemory,
    DoneOrFailedNoretry,
}
fn test_selected_label<Writer: std::io::Write>(
    loaded_devices: LoadedDevices,
    env: &mut ProcessEnv,
    selected_label: usize,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<(Option<LoadedDevices>, TestStatus), Box<dyn std::error::Error>> {
    if env.interactive {
        let mut mode;
        let mut main_code: u8 = 0;
        loop {
            mode = SubprocessMode::NotExeced;
            if let Some(argv0) = &env.argv0 {
                if let Ok(mut child) = std::process::Command::new(argv0)
                    .arg(selected_label.to_string())
                    .arg(env.max_test_bytes.to_string())
                    .spawn()
                {
                    if env.verbose {
                        let _ = writeln!(
                            log_dupler,
                            "Spawned child {child:?} with PID {}",
                            child.id()
                        );
                    }
                    let wait_result = child.wait();
                    let parent_close_requested = close::close_requested();
                    match wait_result {
                        Err(e) => {
                            let _ =
                                writeln!(log_dupler,
                                "wait error: {e}  parent_close_requested: {parent_close_requested}"
                            );
                            return Err("Problem waiting for subprocess".into());
                        }
                        Ok(exit_status) => {
                            if env.verbose {
                                let _ = writeln!(log_dupler, "Subprocess status {exit_status} parent_close_requested {parent_close_requested}");
                            }
                            match exit_status.code() {
                                None => {
                                    return Err("Exit code of test process not available".into())
                                }
                                Some(subprocess_code) => {
                                    main_code = subprocess_code as u8;
                                    let strange_code = (main_code
                                        & close::app_status::SIGNATURE_MASK)
                                        != close::app_status::SIGNATURE;
                                    if strange_code {
                                        let _ = writeln!(
                                            log_dupler,
                                            "Unexpected code {subprocess_code}"
                                        );
                                        return Err(
                                            "Exit code of test process can't be interpreted".into(),
                                        );
                                    }
                                    if main_code
                                        == (close::app_status::SIGNATURE
                                            | close::app_status::RUNTIME_ABORT)
                                    {
                                        mode = SubprocessMode::FailedRetryLowerMemory;
                                    } else {
                                        mode = SubprocessMode::DoneOrFailedNoretry;
                                        if !parent_close_requested
                                            && !close::check_any_bits_set(
                                                main_code,
                                                close::app_status::QUIT_JOB_REQUESTED,
                                            )
                                        {
                                            let _ = writeln!(log_dupler, "Seems child exited for no reason, code {subprocess_code}");
                                            main_code |= close::app_status::RUNTIME_ABORT;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            match mode {
                SubprocessMode::NotExeced => {
                    let _ = writeln!(log_dupler, "Using in-process testing method");
                    break;
                }
                SubprocessMode::FailedRetryLowerMemory => {
                    let smaller_memory = env.max_test_bytes - ALLOCATION_TRY_STEP;
                    if smaller_memory < MIN_WANTED_ALLOCATION {
                        let _ = writeln!(
                            log_dupler,
                            "Using in-process testing method with small memory limit {}",
                            env.max_test_bytes
                        );
                        break;
                    }
                    env.max_test_bytes = smaller_memory;
                    if env.verbose {
                        let _ = writeln!(
                            log_dupler,
                            "retrying subprocess with smaller memory limit {}",
                            env.max_test_bytes
                        );
                    }
                    continue;
                }
                SubprocessMode::DoneOrFailedNoretry => {
                    return Ok((
                        Some(loaded_devices),
                        TestStatus {
                            test_status: main_code
                                | (close::fetch_status() & close::app_status::QUIT_JOB_REQUESTED),
                        },
                    ))
                }
            }
        }
        let _ = writeln!(log_dupler, "Using in-process testing method");
    }
    test_in_this_process(loaded_devices, env, log_dupler)
}
fn init_vk_and_check_errors<Writer: std::io::Write>(
    loaded_devices: LoadedDevices,
    env: &mut ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<(Option<LoadedDevices>, TestStatus), Box<dyn std::error::Error>> {
    if env.device_label.is_none() {
        let LoadedDevices(_, _, _, devices_labeled_from_1) = &loaded_devices;
        let _ = writeln!(log_dupler,);
        for desc in devices_labeled_from_1.iter() {
            let _ = writeln!(log_dupler, "{}", desc.label);
        }
        if env.interactive && devices_labeled_from_1.len() > 1 {
            env.device_label = prompt_for_label(env.verbose);
        } else {
            env.device_label = Some(0usize);
        }
    }
    if env.interactive {
        close::setup_handler(true); //for interactive environments setup handler only after input prompt was run
    }
    if let Some(selected_label) = env.device_label {
        try_fill_default_mem_budget(&loaded_devices, env, log_dupler);

        test_selected_label(loaded_devices, env, selected_label, log_dupler)
    } else {
        Err("Test cancelled, no device selected".into())
    }
}

#[derive(Default)]
struct ProcessEnv {
    argv0: Option<OsString>,
    device_label: Option<usize>,
    max_test_bytes: i64,
    verbose: bool,
    interactive: bool,
}
impl ProcessEnv {
    fn effective_index(&self) -> usize {
        match self.device_label {
            None => 0,
            Some(0) => 0,
            Some(positive) => positive - 1,
        }
    }
}

fn init_running_env() -> ProcessEnv {
    let mut process_env = ProcessEnv {
        verbose: cfg!(feature = "verbose"),
        ..Default::default()
    };
    let mut args_os_iter = std::env::args_os();
    if let Some(argv0) = args_os_iter.next() {
        if let Some(file_stem) = std::path::PathBuf::from(&argv0)
            .file_stem()
            .and_then(|os_str| os_str.to_str())
        {
            process_env.verbose |= file_stem.to_ascii_lowercase().contains("verbose");
        }
        process_env.argv0 = Some(argv0);
        process_env.interactive = true;
        if let Some(argv1_label_str) = args_os_iter
            .next()
            .as_ref()
            .and_then(|os_str| os_str.to_str())
        {
            if let Ok(label_parsed) = argv1_label_str.parse::<usize>() {
                process_env.interactive = false;
                process_env.device_label = Some(label_parsed)
            }
            if let Some(argv2_mem_max_str) = args_os_iter
                .next()
                .as_ref()
                .and_then(|os_str| os_str.to_str())
            {
                if let Ok(mem_max_parsed) = argv2_mem_max_str.parse::<i64>() {
                    process_env.max_test_bytes = mem_max_parsed;
                }
            }
        }
    }
    if process_env.interactive {
        print!("https://github.com/GpuZelenograd/");
        let _ = std::io::stdout().flush();
        let mut color_setter = input::Reader::default();
        color_setter.set_pass_fail_accent_color(false);
        println!(
            "memtest_vulkan v{} by GpuZelenograd",
            env!("CARGO_PKG_VERSION")
        );
        drop(color_setter);
        println!("To finish testing use Ctrl+C");
    }
    process_env
}

fn display_this_process_result(
    maybe_err: Option<Box<dyn std::error::Error>>,
    env: &ProcessEnv,
) -> ! {
    if let Some(e) = maybe_err {
        println!("Runtime error: {e}");
        close::raise_status_bit(close::app_status::RUNTIME_ABORT);
    }
    display_testing_outcome(
        TestStatus {
            test_status: close::fetch_status(),
        },
        env,
    )
}

fn display_testing_outcome(test_status: TestStatus, env: &ProcessEnv) -> ! {
    if !env.interactive {
        close::immediate_exit(false);
    }
    println!();
    let mut key_reader = input::Reader::default();
    let status = test_status.test_status;
    if close::check_any_bits_set(status, close::app_status::QUIT_JOB_REQUESTED) {
        //propagate closing flag to this process, so no risky-during close functions would be used
        close::raise_status_bit(close::app_status::QUIT_JOB_REQUESTED);
    }
    if !close::check_any_bits_set(status, close::app_status::INITED_OK) {
        println!("memtest_vulkan: INIT OR FIRST testing failed due to runtime error");
    } else if close::check_any_bits_set(status, close::app_status::RUNTIME_ABORT)
        && !close::check_any_bits_set(status, close::app_status::RUNTIME_ERRORS)
    {
        println!("memtest_vulkan: First test passed, but THEN runtime error occured");
    } else {
        let has_errors = close::check_any_bits_set(status, close::app_status::RUNTIME_ERRORS);
        if env.interactive {
            key_reader.set_pass_fail_accent_color(has_errors);
        }
        match has_errors {
            true => println!("memtest_vulkan: memory/gpu ERRORS FOUND, testing finished."),
            false => println!("memtest_vulkan: no any errors, testing PASSed."),
        }
    }
    if env.interactive {
        key_reader.wait_any_key();
    }
    drop(key_reader); //restore terminal state before exiting
    close::immediate_exit(false)
}

fn display_result<Writer: std::io::Write>(
    result: Result<(Option<LoadedDevices>, TestStatus), Box<dyn std::error::Error>>,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    match result {
        Ok((_, test_status)) => {
            let _ = log_dupler.flush();
            display_testing_outcome(test_status, env)
        }
        Err(e) => {
            if !env.interactive {
                close::immediate_exit(false);
            }
            println!();
            let mut key_reader = input::Reader::default();
            let _ = writeln!(log_dupler, "memtest_vulkan: early exit during init: {e}");
            let _ = log_dupler.flush();
            if env.interactive {
                key_reader.wait_any_key();
            }
            drop(key_reader); //restore terminal state before exiting
            close::immediate_exit(false)
        }
    }
}

fn main() {
    let mut env = init_running_env();
    if !env.interactive {
        close::setup_handler(false);
    }
    const MAX_LOG_SIZE: u64 = 50 * 1024 * 1024;
    //log is put in current directory. This is intentional - run from other dir to use another log
    let mut log_dupler = output::LogDupler::new(
        std::io::stdout(),
        Some("memtest_vulkan.log".into()),
        MAX_LOG_SIZE,
        if env.interactive {
            "Tester console"
        } else {
            "Tester worker"
        },
    );
    let result = list_devices_ordered_labaled_from_1(env.verbose, &mut log_dupler).and_then(
        |loaded_devices| init_vk_and_check_errors(loaded_devices, &mut env, &mut log_dupler),
    );
    display_result(result, &env, &mut log_dupler);
}
