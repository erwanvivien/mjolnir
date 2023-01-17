use std::{
    io::{BufReader, Cursor},
    path::Path,
};

use gltf::Gltf;
use wgpu::util::DeviceExt;

use crate::{
    model::{self, AnimationClip, Keyframes},
    texture,
};

#[cfg(not(target_arch = "wasm32"))]
const FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"));

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &Path) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let base = reqwest::Url::parse(&format!("{}/", location.origin().unwrap())).unwrap();

    base.join(&file_name.display().to_string()).unwrap()
}

pub async fn load_string(file_name: &Path) -> anyhow::Result<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let url = format_url(file_name);
        let txt = reqwest::get(url).await?.text().await?;

        Ok(txt)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(
            file_name.exists(),
            "Texture file does not exist: {}",
            file_name.display()
        );

        Ok(std::fs::read_to_string(file_name)?)
    }
}

pub async fn load_binary(file_name: &Path) -> anyhow::Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    {
        let url = format_url(file_name);
        let data = reqwest::get(url).await?.bytes().await?.to_vec();

        Ok(data)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(
            file_name.exists(),
            "Texture file does not exist: {}",
            file_name.display()
        );

        Ok(std::fs::read(file_name)?)
    }
}

pub async fn load_texture(
    file_name: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    #[cfg(not(target_arch = "wasm32"))]
    assert!(
        file_name.exists(),
        "Texture file does not exist: {}",
        file_name.display()
    );

    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, &file_name.display().to_string())
}

pub async fn load_model(
    file_name: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<model::Model> {
    #[cfg(not(target_arch = "wasm32"))]
    let file_name = std::path::Path::new(FILE).join("assets").join(file_name);
    #[cfg(target_arch = "wasm32")]
    let file_name = std::path::Path::new("assets").join(file_name);

    log::info!("Loading model: {}", file_name.display());
    if file_name.extension() == Some("obj".as_ref()) {
        load_model_obj(&file_name, device, queue).await
    } else if file_name.extension() == Some("gltf".as_ref()) {
        load_model_gltf(&file_name, device, queue).await
    } else {
        Err(anyhow::anyhow!("Unsupported model format"))
    }
}

pub async fn load_model_obj(
    file_name: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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
            let p = file_name.with_file_name(p);
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = file_name.with_file_name(m.diffuse_texture);
        let diffuse_texture = load_texture(&diffuse_texture, device, queue).await?;

        materials.push(model::Material {
            name: m.name,
            diffuse_texture,
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

    let animations = Vec::new();

    Ok(model::Model {
        meshes,
        materials,
        animations,
    })
}

pub async fn load_model_gltf(
    file_name: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<model::Model> {
    let gltf_text = load_string(file_name).await?;
    let gltf_cursor = Cursor::new(gltf_text);
    let gltf_reader = BufReader::new(gltf_cursor);
    let gltf = Gltf::from_reader(gltf_reader)?;

    // Load buffers
    let mut buffer_data = Vec::new();
    for buffer in gltf.buffers() {
        let bin = match buffer.source() {
            gltf::buffer::Source::Bin => {
                // if let Some(blob) = gltf.blob.as_deref() {
                //     buffer_data.push(blob.into());
                //     println!("Found a bin, saving");
                // };
                Vec::new()
            }
            gltf::buffer::Source::Uri(uri) => {
                let uri = file_name.with_file_name(uri);
                load_binary(&uri).await?
            }
        };

        buffer_data.push(bin);
    }

    // Load animations
    let mut animation_clips = Vec::new();
    for animation in gltf.animations() {
        for channel in animation.channels() {
            let reader = channel.reader(|buffer| Some(&buffer_data[buffer.index()]));
            let timestamps = if let Some(inputs) = reader.read_inputs() {
                match inputs {
                    gltf::accessor::Iter::Standard(times) => {
                        let times: Vec<f32> = times.collect();
                        times
                    }
                    gltf::accessor::Iter::Sparse(_) => {
                        log::error!("Sparse keyframes not supported");
                        Vec::new()
                    }
                }
            } else {
                log::error!("Couldn't read inputs");
                Vec::new()
            };

            let keyframes = if let Some(outputs) = reader.read_outputs() {
                match outputs {
                    gltf::animation::util::ReadOutputs::Translations(translation) => {
                        let translation_vec = translation
                            .map(|tr| {
                                let vector: Vec<f32> = tr.into();
                                vector
                            })
                            .collect();
                        Keyframes::Translation(translation_vec)
                    }
                    _ => todo!(),
                }
            } else {
                log::error!("Couldn't read outputs");
                Keyframes::Other
            };

            animation_clips.push(AnimationClip {
                name: animation.name().unwrap_or("Default").to_string(),
                keyframes,
                timestamps,
            })
        }
    }

    // Load materials
    let mut materials = Vec::new();
    log::info!("Looping through materials");
    for material in gltf.materials() {
        let pbr = material.pbr_metallic_roughness();
        let texture_source = &pbr
            .base_color_texture()
            .map(|tex| tex.texture().source().source())
            .expect("texture");

        let diffuse_texture = match texture_source {
            gltf::image::Source::View { view, .. } => texture::Texture::from_bytes(
                device,
                queue,
                &buffer_data[view.buffer().index()],
                &file_name.display().to_string(),
            )
            .expect("Couldn't load diffuse"),
            gltf::image::Source::Uri { uri, .. } => {
                let uri = file_name.with_file_name(uri);
                load_texture(&uri, device, queue).await?
            }
        };

        materials.push(model::Material {
            name: material.name().unwrap_or("Default Material").to_string(),
            diffuse_texture,
        });
    }

    let mut meshes = Vec::new();

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            log::info!("Node {} {}", node.index(), node.name().unwrap_or("Unnamed"));

            let mesh = node.mesh().expect("Got mesh");
            let primitives = mesh.primitives();
            primitives.for_each(|primitive| {
                let reader = primitive.reader(|buffer| Some(&buffer_data[buffer.index()]));

                log::info!("[START] Reading positions, normals, tex_coords");
                let (positions, normals, tex_coords) = (
                    reader.read_positions().unwrap(),
                    reader.read_normals().unwrap(),
                    reader.read_tex_coords(0).unwrap().into_f32(),
                );
                log::info!("[END  ] Reading positions, normals, tex_coords");

                log::info!("[START] Reading indices");
                let indices = reader.read_indices().map(|indices| indices.into_u32());
                let indices = match indices {
                    Some(indices) => indices.collect::<Vec<_>>(),
                    None => (0..positions.len() as u32).collect(),
                };
                log::info!("[END  ] Reading indices");

                let vertices = positions
                    .zip(normals)
                    .zip(tex_coords)
                    .map(|((position, normal), tex_coords)| model::ModelVertex {
                        position,
                        normal,
                        tex_coords,
                    })
                    .collect::<Vec<_>>();

                log::info!("[START] Creating buffers");
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", file_name)),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Index Buffer", file_name)),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                log::info!("[END  ] Creating buffers");

                meshes.push(model::Mesh {
                    name: file_name.display().to_string(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: indices.len() as u32,
                    // material: m.mesh.material_id.unwrap_or(0),
                    material: 0,
                });
            });
        }
    }

    Ok(model::Model {
        meshes,
        materials,
        animations: animation_clips,
    })
}
