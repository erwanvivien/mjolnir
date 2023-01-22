use std::{collections::HashMap, mem};

use wgpu::{util::DeviceExt, BindGroupLayout, Device, Queue, Surface};

use crate::{
    camera::{Camera, CameraUniform, Projection},
    instance::{Instance, InstanceRaw},
    model::{self, DrawLight, DrawModel, Model, Vertex},
    node::Node,
    particle::ParticleSystem,
    texture,
};

use super::{Pass, UniformPool};

// Global uniform data
// aka camera position and ambient light color
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
    ambient: [f32; 4],
}

// Local uniform data
// aka the individual model's data
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Locals {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub normal: [f32; 4],
    pub lights: [f32; 4],
}

// Uniform for light data (position + color)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    pub color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}

pub struct PhongConfig {
    pub max_lights: usize,
    pub ambient: [u32; 4],
    pub wireframe: bool,
}

pub struct PhongPass {
    // Uniforms
    // pub global_bind_group_layout: BindGroupLayout,
    pub global_uniform_buffer: wgpu::Buffer,
    pub global_bind_group: wgpu::BindGroup,
    pub local_bind_group_layout: BindGroupLayout,
    // pub local_uniform_buffer: wgpu::Buffer,
    local_bind_groups: HashMap<usize, Vec<wgpu::BindGroup>>,
    pub uniform_pool: UniformPool,
    // Textures
    pub depth_texture: texture::Texture,
    // Render pipeline
    pub render_pipeline: wgpu::RenderPipeline,
    // Lighting
    pub light_uniform: LightUniform,
    pub light_buffer: wgpu::Buffer,
    // pub light_bind_group: wgpu::BindGroup,
    pub light_render_pipeline: wgpu::RenderPipeline,
    // Camera
    pub camera_uniform: CameraUniform,
    pub(crate) projection: Projection,
    // Instances
    instance_buffers: HashMap<usize, wgpu::Buffer>,
    light_model: Option<Model>,
}

impl PhongPass {
    const LIGHT_SIZE: wgpu::BufferAddress = mem::size_of::<LightUniform>() as wgpu::BufferAddress;
    const GLOBAL_SIZE: wgpu::BufferAddress = mem::size_of::<Globals>() as wgpu::BufferAddress;
    const LOCAL_SIZE: wgpu::BufferAddress = mem::size_of::<Locals>() as wgpu::BufferAddress;

    pub fn new(
        phong_config: &PhongConfig,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        camera: &Camera,
        light_model: Option<Model>,
    ) -> PhongPass {
        // Setup global uniforms
        // Global bind group layout
        let global_bind_group_layout = {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("[Phong] Globals"),
                entries: &[
                    // Global uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(PhongPass::GLOBAL_SIZE),
                        },
                        count: None,
                    },
                    // Lights
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(PhongPass::LIGHT_SIZE),
                        },
                        count: None,
                    },
                    // Sampler for textures
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
        };

        // Create light uniforms and setup buffer for them
        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };

        let (global_uniform_buffer, light_buffer, global_bind_group) = {
            // Global uniform buffer
            let global_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("[Phong] Globals"),
                size: PhongPass::GLOBAL_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("[Phong] Lights"),
                contents: bytemuck::cast_slice(&[light_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            // We also need a sampler for our textures
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("[Phong] sampler"),
                min_filter: wgpu::FilterMode::Linear,
                mag_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("[Phong] Globals"),
                layout: &global_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: global_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            (global_uniform_buffer, light_buffer, global_bind_group)
        };
        // Combine the global uniform, the lights, and the texture sampler into one bind group

        // Setup local uniforms
        // Local bind group layout
        let local_bind_group_layout = {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("[Phong] Locals"),
                entries: &[
                    // Local uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(PhongPass::LOCAL_SIZE),
                        },
                        count: None,
                    },
                    // Mesh texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            })
        };
        // Setup the render pipeline
        let pipeline_layout = {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("[Phong] Pipeline"),
                bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
                push_constant_ranges: &[],
            })
        };

        let (depth_stencil, primitive, multisample) = {
            // Enable/disable wireframe mode
            let topology = if phong_config.wireframe {
                wgpu::PrimitiveTopology::LineList
            } else {
                wgpu::PrimitiveTopology::TriangleList
            };

            let depth_stencil = Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            });

            // let _color_format = texture::Texture::DEPTH_FORMAT;
            let primitive = wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                topology,
                ..Default::default()
            };
            let multisample = wgpu::MultisampleState {
                ..Default::default()
            };

            (depth_stencil, primitive, multisample)
        };

        let render_pipeline = {
            let vertex_buffers = [model::ModelVertex::desc(), InstanceRaw::desc()];

            // Setup the shader
            // We use specific shaders for each pass to define visual effect
            // and also to have the right shader for the uniforms we pass
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/model.wgsl").into()),
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("[Phong] Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vs_main",
                    buffers: &vertex_buffers,
                },
                primitive,
                depth_stencil: depth_stencil.clone(),
                multisample,
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState {
                            alpha: wgpu::BlendComponent::REPLACE,
                            color: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            })
        };

        // Create depth texture
        let depth_texture = texture::Texture::create_depth_texture(device, config, "depth_texture");

        // Setup camera uniform
        let projection =
            Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(camera, &projection);

        let light_render_pipeline = {
            let light_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/light.wgsl").into()),
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("[Phong] Light Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &light_shader,
                    entry_point: "vs_main",
                    buffers: &[model::ModelVertex::desc()],
                },
                primitive,
                depth_stencil,
                multisample,
                fragment: Some(wgpu::FragmentState {
                    module: &light_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState {
                            alpha: wgpu::BlendComponent::REPLACE,
                            color: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            })
        };

        // Create instance buffer
        let instance_buffers = HashMap::new();
        let uniform_pool = UniformPool::new("[Phong] Locals", PhongPass::LOCAL_SIZE);

        PhongPass {
            // global_bind_group_layout,
            global_uniform_buffer,
            global_bind_group,
            local_bind_group_layout,
            local_bind_groups: Default::default(),
            uniform_pool,
            depth_texture,
            render_pipeline,
            camera_uniform,
            projection,

            light_uniform,
            light_buffer,
            light_render_pipeline,
            instance_buffers,

            light_model,
        }
    }
}

//             render_pass(device, queue, &mut encoder, self, nodes)
fn render_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    phong_pass: &mut PhongPass,
    nodes: &[Node],
    particle_system: &[ParticleSystem],
    view: &wgpu::TextureView,
) {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                // Set the clear color during redraw
                // This is basically a background color applied if an object isn't taking up space
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                }),
                store: true,
            },
        })],
        // Create a depth stencil buffer using the depth texture
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &phong_pass.depth_texture.view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
    });

    // Allocate buffers for local uniforms
    let total_buffers = nodes.len() + particle_system.len();
    if phong_pass.uniform_pool.buffers.len() < total_buffers {
        phong_pass.uniform_pool.alloc_buffers(total_buffers, device);
    }

    // Loop over the nodes/models in a scene and setup the specific models
    // local uniform bind group and instance buffers to send to shader
    // This is separate loop from the render because of Rust ownership
    // (can prob wrap in block instead to limit mutable use)
    // TODO: Change model_index to node_index
    for (model_index, node) in nodes.iter().enumerate() {
        let local_buffer = &phong_pass.uniform_pool.buffers[model_index];

        // We create a bind group for each model's local uniform data
        // and store it in a hash map to look up later
        phong_pass
            .local_bind_groups
            .entry(model_index)
            .or_insert_with(|| {
                #[cfg(debug_assertions)]
                log::debug!("Creating local bind group for model: {}", model_index);
                (0..node.model.materials.len())
                    .map(|mesh_index| {
                        #[cfg(debug_assertions)]
                        log::debug!("Material {}/{}", mesh_index, node.model.materials.len());
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("[Phong] Locals"),
                            layout: &phong_pass.local_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: local_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &node.model.materials[mesh_index].diffuse_texture.view,
                                    ),
                                },
                            ],
                        })
                    })
                    .collect::<Vec<_>>()
            });

        // Setup instance buffer for the model
        // similar process as above using HashMap
        phong_pass
            .instance_buffers
            .entry(model_index)
            .or_insert_with(|| {
                // We condense the matrix properties into a flat array (aka "raw data")
                // (which is how buffers work - so we can "stride" over chunks)
                let instance_data = node
                    .instances
                    .iter()
                    .map(Instance::to_raw)
                    .collect::<Vec<_>>();
                // Create the instance buffer with our data
                let instance_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: bytemuck::cast_slice(&instance_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

                instance_buffer
            });
    }

    for (model_index, particle) in particle_system.iter().enumerate() {
        let model_index = model_index + nodes.len();
        let local_buffer = &phong_pass.uniform_pool.buffers[model_index];

        // We create a bind group for each model's local uniform data
        // and store it in a hash map to look up later
        phong_pass
            .local_bind_groups
            .entry(model_index)
            .or_insert_with(|| {
                log::debug!(
                    "Creating local bind group for particle: {}",
                    model_index - nodes.len()
                );
                (0..particle.model.materials.len())
                    .map(|mesh_index| {
                        #[cfg(debug_assertions)]
                        #[cfg(debug_assertions)]
                        log::debug!("Material {}/{}", mesh_index, particle.model.materials.len());
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("[Phong] Locals"),
                            layout: &phong_pass.local_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: local_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &particle.model.materials[mesh_index].diffuse_texture.view,
                                    ),
                                },
                            ],
                        })
                    })
                    .collect::<Vec<_>>()
            });
    }

    if let Some(light_model) = &phong_pass.light_model {
        // Setup lighting pipeline
        render_pass.set_pipeline(&phong_pass.light_render_pipeline);
        // Draw/calculate the lighting on models
        render_pass.draw_light_model(
            light_model,
            &phong_pass.global_bind_group,
            &phong_pass
                .local_bind_groups
                .get(&0)
                .expect("No local bind group found for lighting")[0],
        );
    }

    // Setup particle pipeline
    render_pass.set_pipeline(&phong_pass.render_pipeline);
    render_pass.set_bind_group(0, &phong_pass.global_bind_group, &[]);
    for (model_index, particle) in particle_system.iter().enumerate() {
        let model_index = model_index + nodes.len();

        // Set the instance buffer unique to the model
        render_pass.set_vertex_buffer(1, particle.instance_buffer.slice(..));

        let model_bind_group = phong_pass.local_bind_groups[&model_index]
            .iter()
            .map(|bind_group| bind_group)
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        log::debug!(
            "[PAR_SYS] Drawing particle_system#{} with {} materials",
            0,
            model_bind_group.len()
        );

        // Draw the model
        render_pass.draw_model_instanced(
            &particle.model,
            0..particle.instances.len() as u32,
            &model_bind_group,
        );
    }

    // Setup render pipeline
    render_pass.set_pipeline(&phong_pass.render_pipeline);
    render_pass.set_bind_group(0, &phong_pass.global_bind_group, &[]);

    // Render/draw all nodes/models
    // We reset index here to use again
    for (model_index, node) in nodes.iter().enumerate() {
        // Set the instance buffer unique to the model
        render_pass.set_vertex_buffer(1, phong_pass.instance_buffers[&model_index].slice(..));

        let model_bind_group = phong_pass.local_bind_groups[&model_index]
            .iter()
            .map(|bind_group| bind_group)
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        log::debug!(
            "[NODE] Drawing model#{} with {} materials",
            model_index,
            &model_bind_group.len()
        );

        // Draw all the model instances
        render_pass.draw_model_instanced(
            &node.model,
            0..node.instances.len() as u32,
            &model_bind_group,
        );
    }
}

impl Pass for PhongPass {
    fn draw(
        &mut self,
        surface: &Surface,
        device: &Device,
        queue: &Queue,
        nodes: &[Node],
        particle_system: &[ParticleSystem],
    ) -> Result<(), wgpu::SurfaceError> {
        let output = surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        render_pass(device, &mut encoder, self, nodes, particle_system, &view);

        queue.submit(Some(encoder.finish()));
        output.present();

        // Since the WGPU breaks return with a Result and error
        // we need to return an `Ok` enum
        Ok(())
    }
}
