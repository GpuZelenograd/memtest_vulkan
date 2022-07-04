// This example currently uses the old and deprecated allocator.
// TODO: Rewrite to use `gpu-alloc`
#![allow(deprecated)]

use erupt::{
    utils::{
        allocator::{Allocator, AllocatorCreateInfo, MemoryTypeFinder},
    },
    vk1_0, DeviceLoader, EntryLoader, InstanceLoader,
};
use std::{
    convert::TryInto,
    ffi::{CStr, CString},
    mem,
};

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

fn main() {
    let entry = EntryLoader::new().unwrap();
    println!(
        "compute example - Vulkan Instance {}.{}.{}",
        vk1_0::version_major(entry.instance_version()),
        vk1_0::version_minor(entry.instance_version()),
        vk1_0::version_patch(entry.instance_version())
    );

    let instance = InstanceLoader::new(&entry, &vk1_0::InstanceCreateInfoBuilder::new(), None).unwrap();

    let (physical_device, queue_family, properties) =
        unsafe { instance.enumerate_physical_devices(None) }
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device, None)
                    .into_iter()
                    .position(|properties| {
                        properties.queue_flags.contains(vk1_0::QueueFlags::COMPUTE)
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let properties = instance.get_physical_device_properties(physical_device, None);
                Some((physical_device, queue_family, properties))
            })
            .max_by_key(|(_, _, properties)| match properties.device_type {
                vk1_0::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk1_0::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .expect("No suitable physical device found");

    println!("Using physical device: {:?}", unsafe {
        CStr::from_ptr(properties.device_name.as_ptr())
    });

    let queue_create_info = vec![vk1_0::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk1_0::PhysicalDeviceFeaturesBuilder::new();

    let create_info = vk1_0::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_create_info)
        .enabled_features(&features);

    let device = DeviceLoader::new(&instance, physical_device, &create_info, None).unwrap();
    let queue = unsafe { device.get_device_queue(queue_family, 0, None) };

    let mut allocator =
        Allocator::new(&instance, physical_device, AllocatorCreateInfo::default()).unwrap();

    let data = Buffer {
        data: (0..21)
            .map(|i| i as f32)
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
            .unwrap(),
    };
    let data_size = mem::size_of_val(&data) as vk1_0::DeviceSize;

    let create_info = vk1_0::BufferCreateInfoBuilder::new()
        .sharing_mode(vk1_0::SharingMode::EXCLUSIVE)
        .usage(vk1_0::BufferUsageFlags::STORAGE_BUFFER)
        .size(data_size);

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

    let desc_pool_sizes = &[vk1_0::DescriptorPoolSizeBuilder::new()
        .descriptor_count(1)
        ._type(vk1_0::DescriptorType::STORAGE_BUFFER)];
    let desc_pool_info = vk1_0::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(desc_pool_sizes)
        .max_sets(1);
    let desc_pool = unsafe { device.create_descriptor_pool(&desc_pool_info, None, None) }.unwrap();

    let desc_layout_bindings = &[vk1_0::DescriptorSetLayoutBindingBuilder::new()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk1_0::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk1_0::ShaderStageFlags::COMPUTE)];
    let desc_layout_info =
        vk1_0::DescriptorSetLayoutCreateInfoBuilder::new().bindings(desc_layout_bindings);
    let desc_layout =
        unsafe { device.create_descriptor_set_layout(&desc_layout_info, None, None) }.unwrap();

    let desc_layouts = &[desc_layout];
    let desc_info = vk1_0::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(desc_layouts);
    let desc_set = unsafe { device.allocate_descriptor_sets(&desc_info) }.unwrap()[0];

    unsafe {
        device.update_descriptor_sets(
            &[vk1_0::WriteDescriptorSetBuilder::new()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk1_0::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk1_0::DescriptorBufferInfoBuilder::new()
                    .buffer(*buffer.object())
                    .offset(buffer.region().start)
                    .range(data_size)])],
            &[],
        )
    };

    let pipeline_layout_desc_layouts = &[desc_layout];
    let pipeline_layout_info =
        vk1_0::PipelineLayoutCreateInfoBuilder::new().set_layouts(pipeline_layout_desc_layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None, None) }.unwrap();

    let spv_code = Vec::from(SHADER);
    let create_info = vk1_0::ShaderModuleCreateInfoBuilder::new().code(&spv_code);
    let shader_mod = unsafe { device.create_shader_module(&create_info, None, None) }.unwrap();

    let entry_point = CString::new("main").unwrap();
    let shader_stage = vk1_0::PipelineShaderStageCreateInfoBuilder::new()
        .stage(vk1_0::ShaderStageFlagBits::COMPUTE)
        .module(shader_mod)
        .name(&entry_point);

    let pipeline_info = &[vk1_0::ComputePipelineCreateInfoBuilder::new()
        .layout(pipeline_layout)
        .stage(shader_stage.build())];
    let pipeline =
        unsafe { device.create_compute_pipelines(None, pipeline_info, None) }.unwrap()[0];

    let cmd_pool_info = vk1_0::CommandPoolCreateInfoBuilder::new().queue_family_index(queue_family);
    let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None, None) }.unwrap();

    let cmd_buf_info = vk1_0::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .command_buffer_count(1)
        .level(vk1_0::CommandBufferLevel::PRIMARY);
    let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_buf_info) }.unwrap()[0];

    let begin_info = vk1_0::CommandBufferBeginInfoBuilder::new()
        .flags(vk1_0::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.begin_command_buffer(cmd_buf, &begin_info).unwrap();
        device.cmd_bind_pipeline(cmd_buf, vk1_0::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd_buf,
            vk1_0::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_dispatch(cmd_buf, 1, 1, 1);
        device.end_command_buffer(cmd_buf).unwrap();
    }

    let fence_info = vk1_0::FenceCreateInfoBuilder::new();
    let fence = unsafe { device.create_fence(&fence_info, None, None) }.unwrap();

    let cmd_bufs = &[cmd_buf];
    let submit_info = &[vk1_0::SubmitInfoBuilder::new().command_buffers(cmd_bufs)];
    unsafe {
        device
            .queue_submit(queue, submit_info, Some(fence))
            .unwrap();
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
    }

    let map = buffer.map(&device, ..data_size).unwrap();
    println!("input: {:?}", data);
    println!("output: {:?}", bytemuck::from_bytes::<Buffer>(map.read()));
    map.unmap(&device).unwrap();

    // Destruction
    unsafe {
        device.device_wait_idle().unwrap();

        allocator.free(&device, buffer);
        device.destroy_pipeline(Some(pipeline), None);
        device.destroy_pipeline_layout(Some(pipeline_layout), None);
        device.destroy_command_pool(Some(cmd_pool), None);
        device.destroy_fence(Some(fence), None);
        device.destroy_descriptor_set_layout(Some(desc_layout), None);
        device.destroy_descriptor_pool(Some(desc_pool), None);
        device.destroy_shader_module(Some(shader_mod), None);
        device.destroy_device(None);

        instance.destroy_instance(None);
    }

    println!("Exited cleanly");
}
