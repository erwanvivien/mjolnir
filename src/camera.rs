#[cfg(target_arch = "wasm")]
use winit::event::ScanCode;
#[cfg(not(target_arch = "wasm"))]
use winit::event::VirtualKeyCode;
use winit::event::{ElementState, KeyboardInput, WindowEvent};

pub struct Camera {
    pub(crate) eye: cgmath::Point3<f32>,
    pub(crate) target: cgmath::Point3<f32>,
    pub(crate) up: cgmath::Vector3<f32>,
    pub(crate) aspect: f32,
    pub(crate) fovy: f32,
    pub(crate) znear: f32,
    pub(crate) zfar: f32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        proj * view
    }
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = (OPENGL_TO_WGPU_MATRIX * camera.build_view_projection_matrix()).into();
    }
}

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    pub(crate) fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    pub(crate) fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        #[cfg(target_arch = "wasm")]
                        scancode,
                        #[cfg(not(target_arch = "wasm"))]
                        virtual_keycode,
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                let mut res = false;

                // Web
                #[cfg(target_arch = "wasm")]
                {
                    log::info!("Current scancode is: {}", &scancode);
                    // 'Z'
                    if scancode == &0x5A {
                        self.is_forward_pressed = is_pressed;
                        res = true;
                    }
                    // 'Q'
                    if scancode == &0x51 {
                        self.is_left_pressed = is_pressed;
                        res = true;
                    }
                    // 'S'
                    if scancode == &0x53 {
                        self.is_backward_pressed = is_pressed;
                        res = true;
                    }
                    // 'D'
                    if scancode == &0x44 {
                        self.is_right_pressed = is_pressed;
                        res = true;
                    }
                }
                #[cfg(not(target_arch = "wasm"))]
                {
                    if virtual_keycode.is_none() {
                        return false;
                    }
                    let key = virtual_keycode.unwrap();
                    log::info!("Current key is: {:?}", &key);

                    if key == VirtualKeyCode::Z || key == VirtualKeyCode::Up {
                        self.is_forward_pressed = is_pressed;
                        res = true;
                    }
                    if key == VirtualKeyCode::Q || key == VirtualKeyCode::Left {
                        self.is_left_pressed = is_pressed;
                        res = true;
                    }
                    if key == VirtualKeyCode::S || key == VirtualKeyCode::Back {
                        self.is_backward_pressed = is_pressed;
                        res = true;
                    }
                    if key == VirtualKeyCode::D || key == VirtualKeyCode::Right {
                        self.is_right_pressed = is_pressed;
                        res = true;
                    }
                }

                res
            }
            _ => false,
        }
    }

    pub(crate) fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = camera.up.cross(forward_norm);

        // Redo radius calc in case the fowrard/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}
