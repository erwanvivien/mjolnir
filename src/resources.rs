use std::io::{BufReader, Cursor};
use std::path::Path;

use wgpu::util::DeviceExt;

use crate::{model, texture};

#[cfg(not(target_arch = "wasm32"))]
const FILE: &'static str = concat!(env!("CARGO_MANIFEST_DIR"));

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &Path) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let base = reqwest::Url::parse(&format!(
        "{}/{}/",
        location.origin().unwrap(),
        option_env!("RES_PATH").unwrap_or("res"),
    ))
    .unwrap();

    base.join(&file_name.display().to_string()).unwrap()
}

#[cfg(target_arch = "wasm32")]
pub async fn load_string(file_name: &Path) -> anyhow::Result<String> {
    let url = format_url(file_name);
    let txt = reqwest::get(url).await?.text().await?;

    Ok(txt)
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn load_string(file_name: &Path) -> anyhow::Result<String> {
    let path = std::path::Path::new(FILE).join("res").join(file_name);
    let txt = std::fs::read_to_string(&path)?;

    Ok(txt)
}

#[cfg(target_arch = "wasm32")]
pub async fn load_binary(file_name: &Path) -> anyhow::Result<Vec<u8>> {
    let url = format_url(file_name);
    let data = reqwest::get(url).await?.bytes().await?.to_vec();

    Ok(data)
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn load_binary(file_name: &Path) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new(FILE).join("res").join(file_name);
    let data = std::fs::read(path)?;

    Ok(data)
}

pub async fn load_texture(
    file_name: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, &file_name.display().to_string())
}

pub async fn load_model(
    file_name: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let path = file_name.clone();
            path.to_owned().set_file_name(p);

            let mat_text = load_string(&path).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture_path = file_name.clone();
        diffuse_texture_path
            .to_owned()
            .set_file_name(m.diffuse_texture);

        let diffuse_texture = load_texture(&diffuse_texture_path, device, queue).await?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: None,
        });

        materials.push(model::Material {
            name: m.name,
            diffuse_texture,
            bind_group,
        })
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: file_name.display().to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}
