// This example currently uses the old and deprecated allocator.
// TODO: Rewrite to use `gpu-alloc`

use erupt::{
    vk, DeviceLoader, EntryLoader, InstanceLoader,
    extensions::ext_debug_utils,
    cstr,
};
use gpu_alloc::{Config, GpuAllocator, Request, UsageFlags};
use gpu_alloc_erupt::{device_properties, EruptMemoryDevice};
use std::{
    convert::TryInto,
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
    mem, 
};

const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");
const SHADER: &[u32] = inline_spirv::inline_spirv!(r#"
#version 460
#extension GL_ARB_separate_shader_objects : enable

#define SIZE 32

layout(local_size_x = SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) restrict buffer Buffer {
    float buf_data[SIZE];
};

void main() {
    buf_data[gl_LocalInvocationIndex] = sqrt(buf_data[gl_LocalInvocationIndex]);
}
"#, glsl, comp);


#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Buffer {
    data: [f32; 21],
}

unsafe impl bytemuck::Zeroable for Buffer {}
unsafe impl bytemuck::Pod for Buffer {}

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
        "Running https://github.com/galkinvv/memtest-vulkan on Vulkan Instance {}.{}.{}",
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

    let instance_create_info = vk::InstanceCreateInfoBuilder::new()
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);


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

    let queue_create_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeaturesBuilder::new();

    let device_create_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_create_info)
        .enabled_features(&features)
        .enabled_layer_names(&device_layers);

    let device = unsafe{ DeviceLoader::new(&instance, physical_device, &device_create_info)}?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let data = Buffer {
        data: (0..21)
            .map(|i| i as f32)
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()?,
    };
    let data_size = mem::size_of_val(&data) as vk::DeviceSize;

    let buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(data_size);

    /*
    let mut allocator =
        Allocator::new(&instance, physical_device, AllocatorCreateInfo::default()).unwrap();

    let buffer = allocator
        .allocate(
            &device,
            unsafe { device.create_buffer(&create_info, None, None) }.unwrap(),
            MemoryTypeFinder::dynamic(),
        )
        .unwrap();
    let mut map = buffer.map(&device, ..data_size).unwrap();
    map.import(bytemuck::bytes_of(&data));
    map.unmap(&device).unwrap();
    */

    let desc_pool_sizes = &[vk::DescriptorPoolSizeBuilder::new()
        .descriptor_count(1)
        ._type(vk::DescriptorType::STORAGE_BUFFER)];
    let desc_pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(desc_pool_sizes)
        .max_sets(1);
    let desc_pool = unsafe { device.create_descriptor_pool(&desc_pool_info, None) }.unwrap();

    let desc_layout_bindings = &[vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let desc_layout_info =
        vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(desc_layout_bindings);
    let desc_layout =
        unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }.unwrap();

    let desc_layouts = &[desc_layout];
    let desc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(desc_layouts);
    let desc_set = unsafe { device.allocate_descriptor_sets(&desc_info) }.unwrap()[0];

    /*
    unsafe {
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSetBuilder::new()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                    .buffer(*buffer.object())
                    .offset(buffer.region().start)
                    .range(data_size)])],
            &[],
        )
    };
    */

    let pipeline_layout_desc_layouts = &[desc_layout];
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(pipeline_layout_desc_layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap();

    let spv_code = Vec::from(SHADER);
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv_code);
    let shader_mod = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

    let entry_point = CString::new("main")?;
    let shader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
        .stage(vk::ShaderStageFlagBits::COMPUTE)
        .module(shader_mod)
        .name(&entry_point);

    let pipeline_info = &[vk::ComputePipelineCreateInfoBuilder::new()
        .layout(pipeline_layout)
        .stage(*shader_stage)];
    let pipeline =
        unsafe { device.create_compute_pipelines(Default::default(), pipeline_info, None) }.unwrap()[0];

    let cmd_pool_info = vk::CommandPoolCreateInfoBuilder::new().queue_family_index(queue_family);
    let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None) }.unwrap();

    let cmd_buf_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY);
    let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_buf_info) }.unwrap()[0];

    let begin_info = vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.begin_command_buffer(cmd_buf, &begin_info).unwrap();
        device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd_buf,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_dispatch(cmd_buf, 1, 1, 1);
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
    }
    /*
    let map = buffer.map(&device, ..data_size).unwrap();
    println!("input: {:?}", data);
    println!("output: {:?}", bytemuck::from_bytes::<Buffer>(map.read()));
    map.unmap(&device).unwrap();
    */
    // Destruction
    unsafe {
        device.device_wait_idle().unwrap();

        //allocator.free(&device, buffer);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_fence(fence, None);
        device.destroy_descriptor_set_layout(desc_layout, None);
        device.destroy_descriptor_pool(desc_pool, None);
        device.destroy_shader_module(shader_mod, None);
        device.destroy_device(None);

        instance.destroy_debug_utils_messenger_ext(messenger, None);
        instance.destroy_instance(None);
    }

    println!("Exited cleanly");
    Ok(())
}
