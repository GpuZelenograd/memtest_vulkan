mod input;

use byte_strings::c_str;
use core::cmp::{max, min};
use erupt::{
    extensions::{ext_debug_utils, ext_memory_budget, ext_pci_bus_info},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use std::{
    ffi::{c_void, CStr},
    fmt,
    io::Write,
    mem,
    sync::{
        atomic::{AtomicBool, Ordering::SeqCst},
        Arc,
    },
    time,
};

const LAYER_KHRONOS_VALIDATION: &CStr = c_str!("VK_LAYER_KHRONOS_validation");
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

@compute @workgroup_size(64, 1, 1)
fn read(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY: u32 = 0x20000u;//don't inner-multiply by window size
    let addr_mod = global_invocation_id[0] % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let new_mod = (global_invocation_id[0] + io.calc_param) % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let effective_addr = global_invocation_id[0] - addr_mod + new_mod; //make read order a bit rotated, not strictly sequential
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
            atomicMax(&io.idx_max, effective_addr * 4u + i);
            atomicMin(&io.idx_min, effective_addr * 4u + i);
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
    //make global_invocation_id processing specific memory addr different on writing compared to reading
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 0x20000u;//don't inner-multiply by window size
    let proccessed_mod = global_invocation_id[0] % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = global_invocation_id[0] + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
    test[proccessed_idx] = test_value_by_index(proccessed_idx);
}

@compute @workgroup_size(64, 1, 1)
fn emulate_write_bugs(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 0x20000u;//don't inner-multiply by window size
    let proccessed_mod = global_invocation_id[0] % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = global_invocation_id[0] + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
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
const TEST_WINDOW_SIZE_GRANULARITY: i64 = VEC_SIZE as i64 * WG_SIZE * ELEMENT_SIZE * 0x20000 as i64;
const TEST_WINDOW_MAX_SIZE: i64 = 4 * 1024 * 1024 * 1024 - TEST_WINDOW_SIZE_GRANULARITY;
const TEST_DATA_KEEP_FREE: i64 = 400 * 1024 * 1024;

#[derive(Default)]
struct U64HexDebug(i64);

impl fmt::Debug for U64HexDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

#[derive(Copy, Clone)]
struct MostlyZeroArr<const LEN: usize>([u32; LEN]);

impl<const LEN: usize> fmt::Display for MostlyZeroArr<LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if LEN < 8
        {
            for i in 0..LEN {
                write!(f, "0x{:08X} ", self.0[i])?;
            }
            return Ok(())
        }
        let mut zero_count = 0;
        for i in 0..LEN {
            let vali = self.0[i];
            if vali != 0 {
                write!(f, "[{}]={},", i, vali)?;
            } else {
                zero_count += 1;
            }
        }
        write!(f, "{} ZEROs", zero_count)
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
        writeln!(f, "err_bit1_idx: {} ", self.err_bit1_idx)?;
        writeln!(f, "err_bitcount: {} ", self.err_bitcount)?;
        writeln!(f, "mem_bitcount: {} ", self.mem_bitcount)?;
        writeln!(f, "actual_ff: {} actual_max: 0x{:08X} actual_min: 0x{:08X} done_iter_or_err:{} iter:{} calc_param 0x{:08X} idxs:{}-{}",
                self.actual_ff, self.actual_max, self.actual_min, self.done_iter_or_err, self.iter, self.calc_param, self.idx_min, self.idx_max)?;
        writeln!(f, "first_elem: {} ", self.first_elem)?;
        Ok(())
    }
}


impl IOBuf {
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
    fn get_error_addresses(
        &self,
        buf_offset: i64,
    ) -> Option<std::ops::RangeInclusive<U64HexDebug>> {
        if self.done_iter_or_err == self.iter {
            None
        } else {
            Some(std::ops::RangeInclusive::<U64HexDebug>::new(
                U64HexDebug(buf_offset + self.idx_min as i64 * ELEMENT_SIZE),
                U64HexDebug(buf_offset + (self.idx_max + 1) as i64 * ELEMENT_SIZE - 1),
            ))
        }
    }
    fn check_vec_first(&self) -> Result<(), Box<dyn std::error::Error>>
    {
        const TEST_IDX: u32 = 1;
        let addr : u32 = TEST_IDX * VEC_SIZE as u32 + self.calc_param + 1u32;
        let shift = addr % 31u32;
        let rotated = addr << shift | addr >> (32 - shift);
        if rotated != self.first_elem.0[0] {
            println!("{} 0x{:08X}", self, rotated);
            return Err("unexpected calculated value, maybe shader execution is broken".into())
        }
        Ok(())
    }
}

trait MapErrStr {
    type ValueType;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
}

impl<T> MapErrStr for erupt::utils::VulkanResult<T> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg =
                res.to_string() + " while getting " + std::any::type_name::<Self::ValueType>();
            msg.to_string().into()
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

fn test_device(
    instance: &erupt::InstanceLoader,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device_create_info: vk::DeviceCreateInfo,
    debug_mode: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let device = unsafe { DeviceLoader::new(&instance, physical_device, &device_create_info) }?;
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let mut budget_structure: ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXT =
        Default::default();
    let mut budget_request = *vk::PhysicalDeviceMemoryProperties2Builder::new();
    budget_request.p_next = &mut budget_structure
        as *mut ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXT
        as *mut c_void;
    let memory_props = unsafe {
        instance.get_physical_device_memory_properties2(physical_device, Some(budget_request))
    }
    .memory_properties;

    let mut allocation_size = 0i64;
    if debug_mode {
        for i in 0..memory_props.memory_type_count as usize {
            println!("memory type flags {:?} heap {}", memory_props.memory_types[i].property_flags, memory_props.memory_types[i].heap_index);
        }
    }
    for i in 0..memory_props.memory_heap_count as usize {
        if debug_mode
        {
            println!("heap size {:4.1}GB budget {:4.1}GB usage {:4.1}GB flags={:#?}",
                memory_props.memory_heaps[i].size as f32/GB,
                budget_structure.heap_budget[i] as f32/GB,
                budget_structure.heap_usage[i] as f32/GB,
                memory_props.memory_heaps[i].flags,
                )
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
        allocation_size = max(allocation_size, heap_free - TEST_DATA_KEEP_FREE);
    }

    let io_data_size = mem::size_of::<IOBuf>() as vk::DeviceSize;

    let io_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(io_data_size);
    let io_buffer = unsafe { device.create_buffer(&io_buffer_create_info, None) }.err_as_str()?;
    let io_mem_reqs = unsafe { device.get_buffer_memory_requirements(io_buffer) };
    let io_mem_index = (0..memory_props.memory_type_count)
        .find(|i| {
            //test buffer comptibility flags expressed as bitmask
            let suitable = (io_mem_reqs.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory_props.memory_types[*i as usize];
            suitable
                && memory_type.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        })
        .ok_or("DEVICE_LOCAL | HOST_COHERENT memory type not available")?;
    if debug_mode {
        println!(
            "IO memory                  type {}: {:?} heap {:?}",
            io_mem_index,
            memory_props.memory_types[io_mem_index as usize],
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
    unsafe { device.bind_buffer_memory(io_buffer, io_memory, 0) }.err_as_str()?;

    let test_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(allocation_size as u64);
    let test_buffer =
        unsafe { device.create_buffer(&test_buffer_create_info, None) }.err_as_str()?;
    let test_mem_reqs = unsafe { device.get_buffer_memory_requirements(test_buffer) };
    allocation_size = test_mem_reqs.size as i64;
    let test_mem_index = (0..memory_props.memory_type_count)
        .find(|i| {
            //test buffer comptibility flags expressed as bitmask
            let suitable = (test_mem_reqs.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory_props.memory_types[*i as usize];
            let memory_heap = memory_props.memory_heaps[memory_type.heap_index as usize];
            suitable
                && memory_heap.size as i64 >= allocation_size
                && memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or("DEVICE_LOCAL test memory type not available")?;

    let test_memory;
    let mut warn_on_budget_alloc_fail = true;
    loop {
        let min_wanted_allocation = TEST_DATA_KEEP_FREE;
        if allocation_size < min_wanted_allocation {
            return Err("can't allocate memory enough for testing".into());
        }

        let test_memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(allocation_size as u64)
            .memory_type_index(test_mem_index);
        let allocated = unsafe { device.allocate_memory(&test_memory_allocate_info, None) };
        if let Some(ok_memory) = allocated.value {
            test_memory = ok_memory;
            break;
        }
        if warn_on_budget_alloc_fail {
            warn_on_budget_alloc_fail = false;
            println!("Failed allocating {:5.1}GB, tryning to use smaller size. More system memory can help.", allocation_size as f32 / GB);
        }
        allocation_size -= min_wanted_allocation;
    }

    if debug_mode {
        println!(
            "Test memory size {:5.1}GB   type {:2}: {:?} {:?}",
            allocation_size as f32 / GB,
            test_mem_index,
            memory_props.memory_types[test_mem_index as usize],
            memory_props.memory_heaps
                [memory_props.memory_types[test_mem_index as usize].heap_index as usize]
        );
    }

    let test_window_count = allocation_size / TEST_WINDOW_MAX_SIZE
        + i64::from(allocation_size % TEST_WINDOW_MAX_SIZE != 0);
    let test_window_count = max(test_window_count, 2); //at least 2 windows: for testing rereads and rws
    let test_window_size = allocation_size / test_window_count;
    let test_window_size = test_window_size - test_window_size % TEST_WINDOW_SIZE_GRANULARITY;
    let test_data_size = test_window_size * test_window_count;

    unsafe { device.destroy_buffer(test_buffer, None) }; //buffer was used for getting memory reuirements. After allocation size may be smaller

    let test_buffer =
        unsafe { device.create_buffer(&test_buffer_create_info.size(test_data_size as u64), None) }
            .err_as_str()?;
    unsafe { device.bind_buffer_memory(test_buffer, test_memory, 0) }.err_as_str()?;

    let desc_pool_sizes = &[vk::DescriptorPoolSizeBuilder::new()
        .descriptor_count(2)
        ._type(vk::DescriptorType::STORAGE_BUFFER)];
    let desc_pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(desc_pool_sizes)
        .max_sets(1);
    let desc_pool = unsafe { device.create_descriptor_pool(&desc_pool_info, None) }.err_as_str()?;

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
    let desc_layout =
        unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }.err_as_str()?;

    let desc_layouts = &[desc_layout];
    let desc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(desc_layouts);
    let desc_set = unsafe { device.allocate_descriptor_sets(&desc_info) }.err_as_str()?[0];

    unsafe {
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSetBuilder::new()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                    .buffer(io_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)])],
            &[],
        );
    }

    let pipeline_layout_desc_layouts = &[desc_layout];
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(pipeline_layout_desc_layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.err_as_str()?;

    let spv_code = Vec::from(READ_SHADER);
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv_code);
    let shader_mod = unsafe { device.create_shader_module(&create_info, None) }.err_as_str()?;

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

    let pipelines_r_w_emul =
        unsafe { device.create_compute_pipelines(Default::default(), &pipeline_infos, None) }
            .err_as_str()?;

    let cmd_pool_info = vk::CommandPoolCreateInfoBuilder::new()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None) }.err_as_str()?;

    let cmd_buf_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY);
    let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_buf_info) }.err_as_str()?[0];

    let test_element_count = (test_window_size / ELEMENT_SIZE) as u32;

    let fence_info = vk::FenceCreateInfoBuilder::new();
    let fence = unsafe { device.create_fence(&fence_info, None) }.err_as_str()?;

    let cmd_bufs = &[cmd_buf];
    let submit_info = &[vk::SubmitInfoBuilder::new().command_buffers(cmd_bufs)];
    let execute_wait_queue =
        |buf_offset: i64, pipeline: vk::Pipeline| -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSetBuilder::new()
                        .dst_set(desc_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(test_buffer)
                            .offset(buf_offset as u64)
                            .range(test_window_size as u64)])],
                    &[],
                );
                device
                    .begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default())
                    .err_as_str()?;
                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );
                device.cmd_dispatch(
                    cmd_buf,
                    test_element_count / WG_SIZE as u32 / VEC_SIZE as u32,
                    1,
                    1,
                );
                device.end_command_buffer(cmd_buf).err_as_str()?;
                device
                    .queue_submit(queue, submit_info, fence)
                    .err_as_str()?;
                device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .err_as_str()?;
                device.reset_fences(&[fence]).err_as_str()?;
                Ok(())
            }
        };
    let iter_count = 100000000; //by default exit after several days of testing
    let mut written_bytes = 0i64;
    let mut read_bytes = 0i64;
    let mut next_report_duration = time::Duration::from_secs(0);
    let mut start = time::Instant::now();
    let stop_requested = Arc::new(AtomicBool::new(false));
    let mut buffer_in = IOBuf::default();
    let mut no_errors = true;
    for iteration in 1..=iter_count {
        buffer_in.prepare_next_iter_write();
        unsafe { std::ptr::write(mapped, buffer_in) }
        let reread_mode_for_win_0 = iteration > 1; //don't write into win 0
        for window_idx in (reread_mode_for_win_0 as i64)..test_window_count {
            let test_offset = test_window_size * window_idx;
            unsafe {
                (*mapped).calc_param = buffer_in.calc_param + window_idx as u32 * 0x81 as u32;
            }
            execute_wait_queue(test_offset, pipelines_r_w_emul[1])?; //use 2 for error simulation
            written_bytes += test_window_size;
        }
        buffer_in.set_calc_param_for_starting_window();
        let mut last_buffer_out: IOBuf = IOBuf::default();
        for window_idx in 0..test_window_count {
            let reread_mode_for_this_win = reread_mode_for_win_0 && window_idx == 0;
            buffer_in.calc_param += window_idx as u32 * 0x81 as u32;
            unsafe {
                std::ptr::write(
                    mapped,
                    if reread_mode_for_this_win {
                        let mut io_buf = IOBuf::default();
                        io_buf.prepare_next_iter_write();
                        io_buf
                    } else {
                        buffer_in
                    },
                );
            }
            let test_offset = test_window_size * window_idx;
            execute_wait_queue(test_offset, pipelines_r_w_emul[0])?;
            read_bytes += test_window_size;
            {
                unsafe {
                    last_buffer_out = std::ptr::read(mapped);
                }
                if let Some(error) = last_buffer_out.get_error_addresses(test_offset) {
                    no_errors = false;
                    println!("{}", last_buffer_out);
                    println!(
                        "Error found. Mode {}, addresses: {:?}",
                        if reread_mode_for_this_win {
                            "NEXT_RE_READ"
                        } else {
                            "INITIAL_READ"
                        },
                        error
                    );
                }
                last_buffer_out.check_vec_first()?;
            }
        }
        let elapsed = start.elapsed();
        let stop_testing = stop_requested.load(SeqCst);
        if elapsed > next_report_duration || stop_testing {
            let passed_secs = elapsed.as_secs_f32();
            let speed_gbps;
            if passed_secs > 0.0001 {
                speed_gbps = (written_bytes + read_bytes) as f32 / GB / passed_secs;
            } else {
                speed_gbps = 0f32;
            }
            println!("{:7} iteration. Since last report passed {:15?} written {:6.1}GB, read: {:6.1}GB   {:6.1}GB/sec", iteration, elapsed, written_bytes as f32 / GB, read_bytes as f32 / GB, speed_gbps);
            if debug_mode {
                println!("{}", last_buffer_out);
            }
            written_bytes = 0i64;
            read_bytes = 0i64;
            let second1 = time::Duration::from_secs(1);
            if next_report_duration.is_zero() {
                next_report_duration = second1; //2nd report after 1 second
                let stop_requested_setter = stop_requested.clone();
                ctrlc::set_handler(move || {
                    println!("   received user interruption, would stop on next iteration");
                    stop_requested_setter.store(true, SeqCst);
                })?;
            } else if next_report_duration == second1 {
                next_report_duration = second1 * 10; //3rd report after 10 seconds
            } else {
                next_report_duration = second1 * 100; //later reports every 100 seconds
            }
            start = time::Instant::now();
        }
        if stop_testing {
            break;
        }
    }
    // Cleanup & Destruction
    unsafe {
        device.device_wait_idle().err_as_str()?;

        device.destroy_buffer(test_buffer, None);
        device.free_memory(test_memory, None);

        device.destroy_buffer(io_buffer, None);
        device.unmap_memory(io_memory);
        device.free_memory(io_memory, None);

        for pipeline in pipelines_r_w_emul {
            device.destroy_pipeline(pipeline, None);
        }
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_fence(fence, None);
        device.destroy_descriptor_set_layout(desc_layout, None);
        device.destroy_descriptor_pool(desc_pool, None);
        device.destroy_shader_module(shader_mod, None);
        device.destroy_device(None);
    }

    Ok(no_errors)
}

fn init_vk_and_check_no_errors(debug_mode: bool) -> Result<bool, Box<dyn std::error::Error>> {
    let entry = EntryLoader::new()?;
    if debug_mode {
        println!(
            "Debug mode enabled ('debug' found in executable), running on Vulkan {}.{}.{}",
            vk::api_version_major(entry.instance_version()),
            vk::api_version_minor(entry.instance_version()),
            vk::api_version_patch(entry.instance_version())
        );
    }
    println!("To finish testing use Ctrl+C");
    println!();

    let mut instance_extensions = Vec::new();
    let mut instance_layers = Vec::new();
    let mut device_layers = Vec::new();

    instance_extensions.push(ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_layers.push(LAYER_KHRONOS_VALIDATION.as_ptr());

    let app_info = vk::ApplicationInfoBuilder::new().api_version(vk::API_VERSION_1_1);
    let instance_create_info = vk::InstanceCreateInfoBuilder::new()
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers)
        .application_info(&app_info);

    let mut messenger = Default::default();
    let instance = unsafe { InstanceLoader::new(&entry, &instance_create_info) }
        .map(|instance| {
            let mut severity = ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                | ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT;
            if debug_mode {
                severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::INFO_EXT;
                //lists all extensions, very verbose
                //severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT;
            }
            let create_info = ext_debug_utils::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                .message_severity(severity)
                .message_type(ext_debug_utils::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(debug_callback));
            messenger = unsafe { instance.create_debug_utils_messenger_ext(&create_info, None) }
                .result()
                .unwrap_or_default();
            device_layers.push(LAYER_KHRONOS_VALIDATION.as_ptr());
            instance
        })
        .or_else(|_| {
            unsafe {
                InstanceLoader::new(
                    &entry,
                    &vk::InstanceCreateInfoBuilder::new().application_info(&app_info),
                )
            } //fallback creation without validation and extensions
        })?;

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
            let mut pci_structure_request = *vk::PhysicalDeviceProperties2Builder::new();
            pci_structure_request.p_next = &mut pci_props_structure
                as *mut ext_pci_bus_info::PhysicalDevicePCIBusInfoPropertiesEXT
                as *mut c_void;

            let properties = instance
                .get_physical_device_properties2(physical_device, Some(pci_structure_request))
                .properties;
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
            ))
        })
        .collect();
    compute_capable_devices.sort_by_key(|(_, _, props, _, pci_props)| {
        let negative_bus_for_reverse_ordering = -(pci_props.pci_bus as i32);
        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => (0, negative_bus_for_reverse_ordering),
            vk::PhysicalDeviceType::INTEGRATED_GPU => (1, negative_bus_for_reverse_ordering),
            _ => (2, negative_bus_for_reverse_ordering),
        }
    });
    let mut numbered_devices = Vec::new();
    for (i, d) in compute_capable_devices.iter().enumerate() {
        let props = d.2;
        let pci_props = d.4;
        let api_info = if debug_mode {
            std::format!(
                "API v.{}.{}.{}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
            )
        } else {
            String::new()
        };
        numbered_devices.push(std::format!(
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
            },
        ));
    }
    for desc in &numbered_devices {
        println!("{desc}");
    }

    let mut device_test_index = Some(0usize);
    let prompt_start = time::Instant::now();
    let mut prompt_duration: Option<time::Duration> = Some(time::Duration::from_secs(10));

    let mut input_reader = input::Reader::default();
    let no_timer_prompt =
        String::from("                                                   Enter index to test:");
    loop {
        let mut prompt = &no_timer_prompt;
        let formatted_prompt: String;
        if let Some(effective_duration) = prompt_duration {
            if effective_duration < prompt_start.elapsed() {
                println!("");
                println!("    ...first device autoselected");
                break;
            } else {
                let duration_left = effective_duration - prompt_start.elapsed();
                formatted_prompt = std::format!(
                    "(first device will be autoselected in {} seconds)   Enter index to test:",
                    duration_left.as_secs()
                );
                prompt = &formatted_prompt;
            }
        }
        match input_reader.input_digit_step(prompt, &time::Duration::from_millis(250))? {
            input::ReaderEvent::Edited => prompt_duration = None,
            input::ReaderEvent::Canceled => {
                println!("");
                device_test_index = None;
                break;
            }
            input::ReaderEvent::Completed => {
                if !input_reader.current_input.is_empty() {
                    match input_reader.current_input.parse::<usize>() {
                        Ok(parsed_idx) => {
                            let mut parsed = parsed_idx;
                            if parsed > 0 {
                                parsed -= 1;
                            }
                            device_test_index = Some(parsed);
                        }
                        Err(_) => {
                            input_reader.current_input.clear();
                            continue;
                        }
                    }
                }
                println!("");
                break;
            }
            input::ReaderEvent::Timeout => {} //just redraw prompt
        }
    }
    drop(input_reader);

    let no_errors;
    if let Some(selected_index) = device_test_index {
        let (physical_device, queue_family, _, _, _) = *(compute_capable_devices
            .get(selected_index)
            .ok_or("No device at given index")?);

        println!("Testing {}", numbered_devices[selected_index]);

        let queue_create_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0])];

        let device_create_info = vk::DeviceCreateInfoBuilder::new()
            .queue_create_infos(&queue_create_info)
            .enabled_layer_names(&device_layers);

        no_errors = test_device(
            &instance,
            physical_device,
            queue_family,
            *device_create_info,
            debug_mode,
        )?;
    } else {
        return Err("Test cancelled, no device selected".into());
    }

    unsafe {
        if !messenger.is_null() {
            instance.destroy_debug_utils_messenger_ext(messenger, None);
        }
        instance.destroy_instance(None);
    }
    Ok(no_errors)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let debug_mode = match std::env::args_os().next() {
        Some(argv0) => match std::path::PathBuf::from(argv0)
            .file_stem()
            .map(std::ffi::OsStr::to_str)
            .flatten()
        {
            Some(file_stem) => file_stem.to_ascii_lowercase().contains("debug"),
            _ => false,
        },
        _ => false,
    };
    print!("https://github.com/GpuZelenograd/");
    let _ = std::io::stdout().flush();
    let mut color_setter = input::Reader::default();
    color_setter.set_pass_fail_accent_color(true);
    println!(
        "memtest_vulkan v{} by GpuZelenograd",
        env!("CARGO_PKG_VERSION")
    );
    drop(color_setter);
    let result = init_vk_and_check_no_errors(debug_mode);
    let mut key_reader = input::Reader::default();
    if let Ok(passed) = result {
        key_reader.set_pass_fail_accent_color(passed);
    }
    match result {
        Ok(false) => println!("memtest_vulkan: memory/GPU errors found, testing finished."),
        Ok(true) => println!("memtest_vulkan: no any errors found, testing finished."),
        Err(e) => println!("memtest_vulkan: testing not done: {}", e),
    }
    key_reader.wait_any_key();
    Ok(())
}
