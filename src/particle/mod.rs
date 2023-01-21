use instant::Duration;

use std::sync::atomic::AtomicU32;

use crate::{instance::Instance, model, pass::phong::Locals};

pub struct ParticleSystem {
    // Local position of model (for relative calculations)
    pub locals: Locals,
    // The vertex buffers and texture data
    pub model: model::Model,
    // An array of positional data for each instance (can just pass 1 instance)
    pub particle_data: Vec<Particle>,

    buffer: wgpu::Buffer,
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
    pub fn new(device: &wgpu::Device, model: model::Model, locals: Locals, count: u32) -> Self {
        use wgpu::util::DeviceExt;

        let particle_data = (0..count)
            .map(|i| Particle {
                position: [i as f32; 3],
                lifetime: 2f32,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        let label = &format!(
            "Particle Instance Buffer {}",
            PARTICLE_SYSTEM_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        );
        #[cfg(not(debug_assertions))]
        let label = "Particle Instance Buffer";

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&particle_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            locals,
            model,
            particle_data,
            buffer,
        }
    }

    pub fn update(&mut self, delta: Duration, queue: &wgpu::Queue) {
        for particle in &mut self.particle_data {
            particle.lifetime = match particle.lifetime {
                life if life < -5f32 => 2f32,
                life => life - delta.as_millis() as f32,
            };
        }

        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.particle_data));
    }

    fn draw(&mut self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {}
}
