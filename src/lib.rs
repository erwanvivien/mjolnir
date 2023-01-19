use std::path::Path;

use ::instant::Instant;
use cgmath::prelude::*;
use context::GraphicsContext;
use node::Node;
use pass::{phong::PhongPass, Pass};

use winit::{dpi::PhysicalPosition, event::*};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod camera;
mod context;
mod instance;
mod instant;
mod model;
mod node;
mod pass;
mod primitives;
mod resources;
mod texture;
mod window;

use crate::{
    camera::{Camera, CameraController},
    model::Keyframes,
    pass::phong::{Locals, PhongConfig},
    window::Window,
};
use crate::{instance::Instance, window::WindowEvents};

pub use std::time::Duration;

struct State {
    ctx: GraphicsContext,
    pass: PhongPass,
    // Window size
    size: winit::dpi::PhysicalSize<u32>,
    // Camera
    camera: Camera,
    camera_controller: CameraController,
    // The 3D models in the scene (as Nodes)
    nodes: Vec<Node>,
    // Animation
    time: Instant,
}

impl State {
    // Initialize the state
    async fn new(window: &Window) -> Self {
        // Save the window size for use later
        let size = window.window.inner_size();

        // Initialize the graphic context
        let ctx = GraphicsContext::new(window).await;

        // Setup the camera and it's initial position
        let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let camera_controller = camera::CameraController::new(4.0, 0.4);

        // Initialize the pass
        let pass_config = PhongConfig {
            max_lights: 1,
            ambient: Default::default(),
            wireframe: false,
        };
        let pass = PhongPass::new(&pass_config, &ctx.device, &ctx.queue, &ctx.config, &camera);

        // Create the 3D objects!
        // Load 3D model from disk or as a HTTP request (for web support)
        let ferris_model = resources::load_model(
            &Path::new("ferris").join("ferris.obj"),
            &ctx.device,
            &ctx.queue,
        )
        .await
        .expect("Couldn't load model. Maybe path is wrong?");

        let car_model = resources::load_model(&Path::new("car.glb"), &ctx.device, &ctx.queue)
            .await
            .unwrap();

        // Create instances for each object with locational data (position + rotation)
        // Renderer currently defaults to using instances. Want one object? Pass a Vec of 1 instance.

        // We create a 2x2 grid of objects by doing 1 nested loop here
        // And use the "displacement" matrix above to offset objects with a gap
        const SPACE_BETWEEN: f32 = 3.0;
        const NUM_INSTANCES_PER_ROW: u32 = 10;
        // More "manual" placement as an example
        let ferris_instances = (0..1)
            .map(|z| {
                let z = SPACE_BETWEEN * (z as f32);
                let position = cgmath::Vector3 { x: z, y: 1.0, z };
                let scale = cgmath::Vector3::new(1f32, 1f32, 1f32);
                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };
                Instance {
                    position,
                    rotation,
                    scale,
                }
            })
            .collect::<Vec<_>>();

        let car_instances = (0..1)
            .map(|z| {
                let z = SPACE_BETWEEN * (z as f32);
                let position = cgmath::Vector3 { x: z, y: 1.0, z };
                let scale = cgmath::Vector3::new(1f32, 1f32, 1f32);
                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };
                Instance {
                    position,
                    rotation,
                    scale,
                }
            })
            .collect::<Vec<_>>();

        let ferris_node = Node {
            parent: 0,
            locals: Locals {
                position: [0f32, 0f32, -0.2f32, 0f32],
                color: [0f32; 4], // Color is not used yet
                normal: [0f32; 4],
                lights: [0f32; 4],
            },
            model: ferris_model,
            instances: ferris_instances,
        };

        let car_node = Node {
            parent: 0,
            locals: Locals {
                position: [0f32; 4],
                color: [0f32; 4], // Color is not used yet
                normal: [0f32; 4],
                lights: [0f32; 4],
            },
            model: car_model,
            instances: car_instances,
        };

        // Put all our nodes into an Vector to loop over later
        let nodes = vec![ferris_node, car_node];

        // Clear color used for mouse input interaction
        let time = Instant::now();

        Self {
            ctx,
            pass,
            size,
            camera,
            camera_controller,
            nodes,
            time,
        }
    }

    // Keeps state in sync with window size when changed
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.ctx.config.width = new_size.width;
            self.ctx.config.height = new_size.height;
            self.ctx
                .surface
                .configure(&self.ctx.device, &self.ctx.config);

            self.pass.projection.resize(new_size.width, new_size.height);

            // Make sure to current window size to depth texture - required for calc
            self.pass.depth_texture = texture::Texture::create_depth_texture(
                &self.ctx.device,
                &self.ctx.config,
                "depth_texture",
            );
        }
    }

    // Handle input using WindowEvent
    pub fn keyboard(&mut self, state: ElementState, keycode: &VirtualKeyCode) -> bool {
        // Send any input to camera controller
        self.camera_controller.process_keyboard(*keycode, state)
    }

    pub fn mouse_moved(&mut self, position: PhysicalPosition<f64>) {
        self.camera_controller.process_mouse(position);
    }

    pub fn mouse_input(&mut self, state: &ElementState, button: &MouseButton) {
        self.camera_controller.process_mouse_input(state, button);
    }

    pub fn scroll(&mut self, delta: &MouseScrollDelta) {
        self.camera_controller.process_scroll(delta);
    }

    fn update(&mut self, dt: Duration) {
        // Sync local app state with camera
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.pass
            .camera_uniform
            .update_view_proj(&self.camera, &self.pass.projection);
        self.ctx.queue.write_buffer(
            &self.pass.global_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.pass.camera_uniform]),
        );

        // Update the light
        let old_position: cgmath::Vector3<_> = self.pass.light_uniform.position.into();
        self.pass.light_uniform.position =
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
                * old_position)
                .into();
        self.ctx.queue.write_buffer(
            &self.pass.light_buffer,
            0,
            bytemuck::cast_slice(&[self.pass.light_uniform]),
        );

        // println!("Time elapsed: {:?}", &self.time.elapsed());

        // Update local uniforms
        let current_time = &self.time.elapsed().as_secs_f32();
        for (node_index, node) in self.nodes.iter_mut().enumerate() {
            // Play animations
            if !node.model.animations.is_empty() {
                // Loop through all animations
                // TODO: Ideally we'd play a certain animation by name - we assume first one for now
                let mut current_keyframe_index = 0;
                for animation in &node.model.animations {
                    for timestamp in &animation.timestamps {
                        if timestamp > current_time {
                            break;
                        }
                        if current_keyframe_index < &animation.timestamps.len() - 1 {
                            current_keyframe_index += 1;
                        }
                    }
                }

                // Update locals with current animation
                let current_animation = &node.model.animations[0].keyframes;
                let mut current_frame: Option<&Vec<f32>> = None;
                match current_animation {
                    Keyframes::Translation(frames) => {
                        current_frame = Some(&frames[current_keyframe_index])
                    }
                    Keyframes::Other => (),
                }

                if let Some(current_frame) = current_frame {
                    node.locals.position = [
                        current_frame[0],
                        current_frame[1],
                        current_frame[2],
                        node.locals.position[3],
                    ];
                }
            }

            self.pass
                .uniform_pool
                .update_uniform(node_index, node.locals, &self.ctx.queue);
        }
    }

    // Primary render flow
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if let Err(err) = self.pass.draw(
            &self.ctx.surface,
            &self.ctx.device,
            &self.ctx.queue,
            &self.nodes,
        ) {
            log::error!("Error in draw: {:?}", err);
        }

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    pub fn init_logs() {
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            use log::LevelFilter::*;
            env_logger::builder()
                .filter_module("mjolnir", Info)
                .filter_module("wgpu_core", Warn)
                .init();
        }
    }

    init_logs();

    let window = Window::new();

    // State::new uses async code, so we're going to wait for it to finish
    let mut app = State::new(&window).await;
    let mut last_render_time = instant::Instant::now(); // NEW!

    // @TODO: Wire up state methods again (like render)
    window.run(move |event| match event {
        WindowEvents::Resized { width, height } => {
            app.resize(winit::dpi::PhysicalSize { width, height });
        }
        WindowEvents::Draw => {
            let dt = last_render_time.elapsed();
            last_render_time = instant::Instant::now();

            app.update(dt);
            if let Err(err) = app.render() {
                log::error!("Error in rendering {:?}", err);
            }
        }
        WindowEvents::Keyboard {
            state,
            virtual_keycode,
        } => {
            app.keyboard(state, virtual_keycode);
        }

        WindowEvents::MouseWheel { delta } => {
            app.scroll(delta);
        }

        WindowEvents::MouseMoved { position } => {
            app.mouse_moved(*position);
        }

        WindowEvents::MouseInput { state, button } => {
            app.mouse_input(state, button);
        }
    });
}
