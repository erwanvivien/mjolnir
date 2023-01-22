use cgmath::{Rotation3, Vector3};
use instant::Duration;

use std::{
    f32::consts::{FRAC_1_PI, FRAC_PI_2, PI},
    sync::atomic::AtomicU32,
};

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
    pub lifetimes: Vec<(f32, f32)>,
    pub instance_buffer: wgpu::Buffer,
}

#[cfg(debug_assertions)]
const PARTICLE_SYSTEM_ID: AtomicU32 = AtomicU32::new(0);

impl ParticleSystem {
    const RADIUS: f32 = 0.3f32;
    const DIAMETER: f32 = Self::RADIUS * 2f32;

    const POS_WHEEL_BACK_LEFT: Vector3<f32> = cgmath::Vector3 {
        x: 0.66f32,
        y: 0f32,
        z: -0.98f32,
    };
    const POS_WHEEL_BACK_RIGHT: Vector3<f32> = cgmath::Vector3 {
        x: -Self::POS_WHEEL_BACK_LEFT.x,
        ..Self::POS_WHEEL_BACK_LEFT
    };

    pub fn new_particle() -> Instance {
        let angle = rand::random::<f32>() * 2f32 * std::f32::consts::PI;
        let (sin, cos) = angle.sin_cos();

        let center = if rand::random::<bool>() {
            Self::POS_WHEEL_BACK_LEFT
        } else {
            Self::POS_WHEEL_BACK_RIGHT
        };

        let position = cgmath::Vector3 {
            x: center.x,
            y: center.y + Self::RADIUS * cos,
            z: center.z + Self::RADIUS * sin,
        };

        // Generate random scale
        let scale = cgmath::Vector3::new(0f32, 0f32, 0f32);
        fn rand_angle() -> cgmath::Rad<f32> {
            cgmath::Rad(rand::random::<f32>() * 2f32 * std::f32::consts::PI)
        }
        let rotation = cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_x(), rand_angle())
            * cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), rand_angle())
            * cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), rand_angle());

        Instance {
            position,
            rotation,
            scale,
        }
    }

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
            lifetimes: vec![(1f32, 1f32); instance_raw.len()],
            instance_buffer,
        }
    }

    pub fn update(&mut self, delta: Duration, queue: &wgpu::Queue) {
        let delta = delta.as_secs_f32();

        for (instance, lifetime) in self.instances.iter_mut().zip(self.lifetimes.iter_mut()) {
            lifetime.0 += delta;
            if lifetime.0 > lifetime.1 {
                let new_lifetime = rand::random::<f32>() * 2f32 + 0.5f32;
                *lifetime = (0f32, new_lifetime);
                *instance = Self::new_particle();
            }

            if instance.position.y < -Self::RADIUS / 2f32 {
                let speed = ((-instance.position.y / 0.6f32 + 0.5f32) * 2f32).powf(2f32) * 0.25f32;
                #[cfg(debug_assertions)]
                assert!(speed >= 0f32 && speed <= 1f32);

                instance.position.z -= delta * speed * 2f32;
            }

            let life_percent = (lifetime.0 / lifetime.1) * PI;
            let scale = life_percent.sin();

            instance.scale = cgmath::Vector3::new(
                scale * 0.015 + 0.01,
                scale * 0.015 + 0.01,
                scale * 0.015 + 0.01,
            );
        }

        let instance_raw = &self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();

        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instance_raw));
    }
}
