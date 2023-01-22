use instant::Duration;

use std::{fmt::format, sync::atomic::AtomicU32};

use crate::{
    instance::{Instance, InstanceRaw},
    model,
    node::Node,
    pass::phong::Locals,
};

pub struct ParticleSystem {
    // ID of parent Node
    pub parent: u32,
    // Local position of model (for relative calculations)
    pub locals: Locals,
    // The vertex buffers and texture data
    pub model: model::Model,
    // An array of positional data for each instance (can just pass 1 instance)
    pub instances: Vec<Instance>,
    pub instance_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    padding_0: f32,
    // pub velocity: [f32; 3],
    /// The lifetime of the particle in milliseconds
    pub lifetime: f32,
    padding_1: [f32; 3],
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: [0f32; 3],
            padding_0: 0f32,
            lifetime: 0f32,
            padding_1: [0f32; 3],
        }
    }
}

#[cfg(debug_assertions)]
const PARTICLE_SYSTEM_ID: AtomicU32 = AtomicU32::new(0);

impl ParticleSystem {
    pub fn new(device: &wgpu::Device, node: Node) -> Self {
        use wgpu::util::DeviceExt;

        let instance_raw = &node
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            #[cfg(debug_assertions)]
            label: Some(&format!(
                "Particle System {}",
                PARTICLE_SYSTEM_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            )),
            #[cfg(not(debug_assertions))]
            label: None,
            contents: bytemuck::cast_slice(instance_raw),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            parent: node.parent,
            locals: node.locals,
            model: node.model,
            instances: node.instances,
            instance_buffer,
        }
    }

    pub fn update(&mut self, delta: Duration, queue: &wgpu::Queue) {
        let delta = delta.as_secs_f32();
        for instance in &mut self.instances {
            instance.scale[0] = (instance.scale[0] + delta).rem_euclid(5f32);
            instance.scale[1] = (instance.scale[1] + delta).rem_euclid(5f32);
            instance.scale[2] = (instance.scale[2] + delta).rem_euclid(5f32);
        }

        let instance_raw = &self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();

        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instance_raw));
    }
}
