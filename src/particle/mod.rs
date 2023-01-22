use cgmath::{Rotation3, Vector3};
use instant::Duration;
use rand_distr::{Distribution, Poisson, WeightedIndex};

use std::f32::consts::PI;
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicU32;

use crate::{
    instance::{Instance, InstanceRaw},
    model,
    node::Node,
    pass::phong::Locals,
};

#[derive(Debug, Clone, Copy)]
pub enum ParticleType {
    Movement(Vector3<f32>),
    Rotation((f32, Vector3<f32>)),
}

pub struct Particle {
    pub instance: Instance,

    pub lifetime: (f32, f32),
    ty: ParticleType,
}

impl Particle {
    fn to_raw(&self) -> InstanceRaw {
        self.instance.to_raw()
    }
}

impl From<Instance> for Particle {
    fn from(instance: Instance) -> Self {
        let direction = Vector3::new(
            (rand::random::<f32>() * 2f32 - 1f32) * 0.2f32,
            (rand::random::<f32>() * 2f32 - 1f32) * 0.2f32,
            -1f32,
        );

        Self {
            instance,
            lifetime: (0f32, rand::random::<f32>()),
            ty: ParticleType::Movement(direction),
        }
    }
}

pub struct ParticleSystem {
    // ID of parent Node
    pub parent: u32,
    // Local position of model (for relative calculations)
    pub locals: Locals,
    // The vertex buffers and texture data
    pub model: model::Model,
    // An array of positional data for each instance (can just pass 1 instance)
    pub instances: Vec<Particle>,
    pub instance_buffer: wgpu::Buffer,
}

#[cfg(debug_assertions)]
#[allow(clippy::declare_interior_mutable_const)]
const PARTICLE_SYSTEM_ID: AtomicU32 = AtomicU32::new(0);

impl ParticleSystem {
    const RADIUS: f32 = 0.3f32;
    const MAX_PARTICLES: usize = 800;

    const POS_WHEEL_BACK_LEFT: Vector3<f32> = cgmath::Vector3 {
        x: 0.66f32,
        y: 0f32,
        z: -0.98f32,
    };
    const POS_WHEEL_BACK_RIGHT: Vector3<f32> = cgmath::Vector3 {
        x: -Self::POS_WHEEL_BACK_LEFT.x,
        ..Self::POS_WHEEL_BACK_LEFT
    };

    pub fn new_particle() -> Particle {
        let direction = Vector3::new(
            (rand::random::<f32>() * 2f32 - 1f32) * 0.2f32,
            (rand::random::<f32>() * 2f32 - 1f32) * 0.2f32,
            -1f32,
        );
        let choices = [{ ParticleType::Movement(direction) }, {
            let poi = Poisson::new(45f32).unwrap();
            let angle = -poi.sample(&mut rand::thread_rng()) / 90f32 * PI;

            let center = if rand::random::<bool>() {
                Self::POS_WHEEL_BACK_LEFT
            } else {
                Self::POS_WHEEL_BACK_RIGHT
            };

            ParticleType::Rotation((angle, center))
        }];
        let weights = [9, 1];
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = rand::thread_rng();

        let ty_index = dist.sample(&mut rng);
        let ty = choices[ty_index];

        let lifetime_factor = match &ty {
            ParticleType::Movement(_) => 0.8f32,
            ParticleType::Rotation(_) => 2f32,
        };

        let center = if rand::random::<bool>() {
            Self::POS_WHEEL_BACK_LEFT
        } else {
            Self::POS_WHEEL_BACK_RIGHT
        };

        let position = match ty {
            ParticleType::Movement(_) => cgmath::Vector3 {
                y: center.y - Self::RADIUS + 0.05f32,
                ..center
            },
            ParticleType::Rotation((angle, center)) => {
                let (sin, cos) = angle.sin_cos();
                cgmath::Vector3 {
                    x: center.x - 0.05f32 * center.x.signum(),
                    y: center.y + Self::RADIUS * cos,
                    z: center.z + Self::RADIUS * sin,
                }
            }
        };

        // Generate random scale
        let scale = cgmath::Vector3::new(0f32, 0f32, 0f32);
        fn rand_angle() -> cgmath::Rad<f32> {
            cgmath::Rad(rand::random::<f32>() * 2f32 * std::f32::consts::PI)
        }
        let rotation = cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_x(), rand_angle())
            * cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), rand_angle())
            * cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), rand_angle());

        let instance = Instance {
            position,
            rotation,
            scale,
        };

        Particle {
            instance,
            lifetime: (0f32, rand::random::<f32>() * lifetime_factor),
            ty,
        }
    }

    pub fn new(device: &wgpu::Device, node: Node) -> Self {
        use wgpu::util::DeviceExt;

        let instances = (0..Self::MAX_PARTICLES)
            .map(|_| Self::new_particle())
            .collect::<Vec<_>>();

        let instance_raw = &instances.iter().map(Particle::to_raw).collect::<Vec<_>>();

        #[allow(clippy::borrow_interior_mutable_const)]
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
            instances,
            instance_buffer,
        }
    }

    pub fn update(&mut self, delta: Duration, queue: &wgpu::Queue) {
        let delta = delta.as_secs_f32();

        for particle in self.instances.iter_mut() {
            let lifetime = particle.lifetime;
            if lifetime.0 > lifetime.1 {
                *particle = Self::new_particle();
            }

            let Particle {
                instance,
                lifetime,
                ty,
            } = particle;
            lifetime.0 += delta;

            match ty {
                ParticleType::Movement(direction) => {
                    let is_back =
                        instance.position.z < Self::POS_WHEEL_BACK_LEFT.z - Self::RADIUS * 1.5f32;
                    let is_bottom = instance.position.y < -Self::RADIUS / 2f32;

                    if is_back {
                        instance.position.y += delta * (-instance.position.z).sqrt();
                    }
                    if is_bottom || is_back {
                        instance.position += *direction * delta * 4f32;
                    }
                    instance.position.y = instance.position.y.max(-Self::RADIUS);

                    // instance.position.z = 100f32;
                }
                ParticleType::Rotation((angle, center)) => {
                    *angle += delta * 8f32; // * ((lifetime.1 - lifetime.0) / lifetime.1);

                    let (sin, cos) = angle.sin_cos();
                    instance.position = cgmath::Vector3 {
                        y: center.y + Self::RADIUS * cos,
                        z: center.z + Self::RADIUS * sin,
                        ..*center
                    };

                    // instance.position.z = 100f32;
                }
            }

            let life_percent = (lifetime.0 / lifetime.1) * PI;
            let scale = life_percent.sin();
            let scale_factor = match ty {
                ParticleType::Movement(_) => 0.05f32,
                ParticleType::Rotation(_) => 0.01f32,
            };
            instance.scale = cgmath::Vector3::new(
                scale * scale_factor + 0.01,
                scale * scale_factor + 0.01,
                scale * scale_factor + 0.01,
            );
        }

        let instance_raw = &self
            .instances
            .iter()
            .map(Particle::to_raw)
            .collect::<Vec<_>>();

        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instance_raw));
    }
}
