use std::path::Path;

use wgpu::util::DeviceExt;

use crate::{
    model::{self, ModelVertex},
    resources::load_texture,
};

use super::PrimitiveMesh;

pub fn plane_vertices(scale: f32) -> Vec<ModelVertex> {
    vec![
        // Front face
        ModelVertex {
            position: [-scale, -scale, scale],
            normal: [0.0, 0.0, 1.0],
            tex_coords: [0.0, 0.0],
        },
        ModelVertex {
            position: [scale, -scale, scale],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 0.0],
        },
        ModelVertex {
            position: [scale, scale, scale],
            normal: [1.0, 0.0, 0.0],
            tex_coords: [1.0, 1.0],
        },
        ModelVertex {
            position: [-scale, scale, scale],
            normal: [-1.0, 0.0, 0.0],
            tex_coords: [0.0, 1.0],
        },
    ]
}

pub fn plane_indices() -> Vec<u32> {
    vec![
        0, 1, 2, 0, 2, 3, // front
    ]
}

pub async fn new(device: &wgpu::Device, queue: &wgpu::Queue, scale: f32) -> PrimitiveMesh {
    let primitive_type = "Plane";

    log::info!("[PRIMITIVE] Creating plane materials");
    // Setup materials
    // We can't have empty material (since shader relies o n bind group)
    // And it doesn't accept Option/None, so we give it a placeholder image
    let mut materials = Vec::new();
    let diffuse_texture = load_texture(
        &Path::new("assets").join("default_texture.png"),
        device,
        queue,
    )
    .await
    .expect("Couldn't load placeholder texture for primitive");

    materials.push(model::Material {
        name: primitive_type.to_string(),
        diffuse_texture,
    });

    log::info!("[PRIMITIVE] Creating plane mesh buffers");
    let mut meshes = Vec::new();

    let vertices = plane_vertices(scale);
    let indices = plane_indices();

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{:?} Vertex Buffer", primitive_type)),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{:?} Index Buffer", primitive_type)),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    meshes.push(model::Mesh {
        name: primitive_type.to_string(),
        vertex_buffer,
        index_buffer,
        num_elements: indices.len() as u32,
        material: 0,
    });

    let animations = Vec::new();

    let model = model::Model {
        meshes,
        materials,
        animations,
    };

    PrimitiveMesh { model }
}
