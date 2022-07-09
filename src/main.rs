use erupt::{
    vk, DeviceLoader, EntryLoader, InstanceLoader,
    extensions::ext_debug_utils,
    cstr,
};
use std::{
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
    mem, 
};

const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");
const READ_SHADER: &[u32] = memtest_vulkan_build::compiled_vk_compute_spirv!(r#"

struct IOBuffer
{
    max: u32,
    min: u32,
    sum: u32,
    write_count: u32,
    read_count: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: IOBuffer;
@group(0) @binding(1) var<storage, read_write> prepared_data: array<u32>;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if (buf.write_count > 0)
    {
        prepared_data[global_invocation_id[0]] = global_invocation_id[0];
    }
    if (buf.read_count > 0)
    {
        atomicMax(&buf.max, prepared_data[global_invocation_id[0]]);
        atomicMin(&buf.min, prepared_data[global_invocation_id[0]]);
        atomicAdd(&buf.sum, arrayLength(&prepared_data)/32u);
    }
}
"#);

const WG_SIZE: u64 = 32;
const ELEMENT_SIZE: u64 = std::mem::size_of::<u32>() as u64;

#[derive(Debug)]
#[repr(C)]
struct IOBuffer
{
    max: u32,
    min: u32,
    sum: u32,
    write_count: u32,
    read_count: u32,
}

impl Default for IOBuffer
{
    fn default() -> Self { 
        IOBuffer { max : u32::MIN, min : u32::MAX, sum : 0, write_count : 0, read_count: 0}
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


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let entry = EntryLoader::new()?;
    println!(
        "Running https://github.com/galkinvv/memtest_vulkan on Vulkan Instance {}.{}.{}",
        vk::api_version_major(entry.instance_version()),
        vk::api_version_minor(entry.instance_version()),
        vk::api_version_patch(entry.instance_version())
    );

    let mut instance_extensions = Vec::new();
    let mut instance_layers = Vec::new();
    let mut device_layers = Vec::new();

    instance_extensions.push(ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_layers.push(LAYER_KHRONOS_VALIDATION);
    device_layers.push(LAYER_KHRONOS_VALIDATION);

    let app_info = vk::ApplicationInfoBuilder::new().
        api_version(vk::API_VERSION_1_1);
    let instance_create_info = vk::InstanceCreateInfoBuilder::new()
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers)
        .application_info(&app_info);

    let instance = unsafe { InstanceLoader::new(&entry, &instance_create_info)}?;

    let messenger = {
        let create_info = ext_debug_utils::DebugUtilsMessengerCreateInfoEXTBuilder::new()
            .message_severity(
                ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                    | ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
                //ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
            )
            .message_type(
                ext_debug_utils::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                    | ext_debug_utils::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                    | ext_debug_utils::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
            )
            .pfn_user_callback(Some(debug_callback));

        unsafe { instance.create_debug_utils_messenger_ext(&create_info, None) }.unwrap()
    };


    let (physical_device, queue_family, properties) =
        unsafe { instance.enumerate_physical_devices(None) }
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device, None)
                    .into_iter()
                    .position(|properties| {
                        properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let properties = instance.get_physical_device_properties(physical_device);
                Some((physical_device, queue_family, properties))
            })
            .max_by_key(|(_, _, properties)| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .expect("No suitable physical device found");

    println!("Using physical device: {:?}", unsafe {
        CStr::from_ptr(properties.device_name.as_ptr())
    });

    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let queue_create_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeaturesBuilder::new();

    let device_create_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_create_info)
        .enabled_features(&features)
        .enabled_layer_names(&device_layers);

    let device = unsafe { DeviceLoader::new(&instance, physical_device, &device_create_info)}?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let io_data_size = mem::size_of::<IOBuffer>() as vk::DeviceSize;

    let io_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(io_data_size);
    let io_buffer = unsafe {device.create_buffer(&io_buffer_create_info, None)}.unwrap();
    let io_mem_reqs = unsafe {device.get_buffer_memory_requirements(io_buffer)};
    let io_mem_index = (0..memory_props.memory_type_count)
        .find(|i| {
            //test buffer comptibility flags expressed as bitmask
            let suitable = (io_mem_reqs.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory_props.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        }).ok_or("DEVICE_LOCAL | HOST_COHERENT memory type not available")?;
    println!("IO memory type: index {}: {:?} heap {:?}", io_mem_index, memory_props.memory_types[io_mem_index as usize], memory_props.memory_heaps[memory_props.memory_types[io_mem_index as usize].heap_index as usize]);

    let io_memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(io_mem_reqs.size)
        .memory_type_index(io_mem_index);
    let io_memory = unsafe{device.allocate_memory(&io_memory_allocate_info, None)}.unwrap();

    let mapped: *mut IOBuffer = unsafe{mem::transmute(device.map_memory(io_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::default()).unwrap())};
    unsafe{device.bind_buffer_memory(io_buffer, io_memory, 0)}.unwrap();

    let test_data_size = 2*1024*1024*1024 - WG_SIZE * ELEMENT_SIZE;

    let test_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(test_data_size);
    let test_buffer = unsafe {device.create_buffer(&test_buffer_create_info, None)}.unwrap();
    let test_mem_reqs = unsafe {device.get_buffer_memory_requirements(test_buffer)};
    let test_mem_index = (0..memory_props.memory_type_count)
        .find(|i| {
            //test buffer comptibility flags expressed as bitmask
            let suitable = (test_mem_reqs.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory_props.memory_types[*i as usize];
            let memory_heap = memory_props.memory_heaps[memory_type.heap_index as usize];
            suitable && memory_heap.size >= test_data_size && memory_type.property_flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL)
        }).ok_or("DEVICE_LOCAL test memory type not available")?;
    println!("Test memory type: index {}: {:?} heap {:?}", test_mem_index, memory_props.memory_types[test_mem_index as usize], memory_props.memory_heaps[memory_props.memory_types[test_mem_index as usize].heap_index as usize]);

    let test_memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(test_mem_reqs.size)
        .memory_type_index(test_mem_index);
    let test_memory = unsafe{device.allocate_memory(&test_memory_allocate_info, None)}.unwrap();
    unsafe{device.bind_buffer_memory(test_buffer, test_memory, 0)}.unwrap();

    let desc_pool_sizes = &[vk::DescriptorPoolSizeBuilder::new()
        .descriptor_count(2)
        ._type(vk::DescriptorType::STORAGE_BUFFER)];
    let desc_pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(desc_pool_sizes)
        .max_sets(1);
    let desc_pool = unsafe { device.create_descriptor_pool(&desc_pool_info, None) }.unwrap();

    let desc_layout_bindings = &[vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::COMPUTE),

        vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        ];
    let desc_layout_info =
        vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(desc_layout_bindings);
    let desc_layout =
        unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }.unwrap();

    let desc_layouts = &[desc_layout];
    let desc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(desc_layouts);
    let desc_set = unsafe { device.allocate_descriptor_sets(&desc_info) }.unwrap()[0];

    unsafe {
        device.update_descriptor_sets(
            &[
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                    .buffer(io_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]),
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                    .buffer(test_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]),
            ],
            &[],
        )
    };

    let pipeline_layout_desc_layouts = &[desc_layout];
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(pipeline_layout_desc_layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap();

    let read_spv_code = Vec::from(READ_SHADER);
    let read_create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&read_spv_code);
    let read_shader_mod = unsafe { device.create_shader_module(&read_create_info, None) }.unwrap();

    let entry_point = CString::new("main")?;
    let read_shader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
        .stage(vk::ShaderStageFlagBits::COMPUTE)
        .module(read_shader_mod)
        .name(&entry_point);

    let pipeline_info = &[vk::ComputePipelineCreateInfoBuilder::new()
        .layout(pipeline_layout)
        .stage(*read_shader_stage)];
    let pipeline =
        unsafe { device.create_compute_pipelines(Default::default(), pipeline_info, None) }.unwrap()[0];

    let cmd_pool_info = vk::CommandPoolCreateInfoBuilder::new().queue_family_index(queue_family);
    let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None) }.unwrap();

    let cmd_buf_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY);
    let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_buf_info) }.unwrap()[0];

    unsafe {
        let buffer_in = &mut *mapped;
        *buffer_in = IOBuffer::default();
        println!("input: {:?}", buffer_in);
        buffer_in.write_count = 1;
    }

    unsafe {
        device.begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default()).unwrap();
        device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd_buf,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_dispatch(cmd_buf, (test_data_size/WG_SIZE/ELEMENT_SIZE) as u32, 1, 1);
        device.end_command_buffer(cmd_buf).unwrap();
    }

    let fence_info = vk::FenceCreateInfoBuilder::new();
    let fence = unsafe { device.create_fence(&fence_info, None) }.unwrap();

    let cmd_bufs = &[cmd_buf];
    let submit_info = &[vk::SubmitInfoBuilder::new().command_buffers(cmd_bufs)];
    unsafe {
        device
            .queue_submit(queue, submit_info, fence)
            .unwrap();
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        device.reset_fences(&[fence]).unwrap();
    }
    unsafe {
        let buffer_in_out = &mut *mapped;
        buffer_in_out.write_count = 0;
        buffer_in_out.read_count = 1;
        println!("medium: {:?}", buffer_in_out);
    }
    unsafe {
        device
            .queue_submit(queue, submit_info, fence)
            .unwrap();
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        device.reset_fences(&[fence]).unwrap();
    }
    
    unsafe {
        let buffer_out = &*mapped;
        println!("output: {:?}", buffer_out);
    }
    
    // Cleanup & Destruction
    unsafe {
        device.device_wait_idle().unwrap();

        device.destroy_buffer(test_buffer, None);
        device.free_memory(test_memory, None);

        device.destroy_buffer(io_buffer, None);
        device.unmap_memory(io_memory);
        device.free_memory(io_memory, None);

        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_fence(fence, None);
        device.destroy_descriptor_set_layout(desc_layout, None);
        device.destroy_descriptor_pool(desc_pool, None);
        device.destroy_shader_module(read_shader_mod, None);
        device.destroy_device(None);

        instance.destroy_debug_utils_messenger_ext(messenger, None);
        instance.destroy_instance(None);
    }

    println!("Exited cleanly");
    Ok(())
}
