use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window,
};

pub enum WindowEvents<'a> {
    Resized {
        width: u32,
        height: u32,
    },
    Keyboard {
        state: ElementState,
        virtual_keycode: &'a VirtualKeyCode,
    },
    MouseInput {
        state: &'a ElementState,
        button: &'a MouseButton,
    },
    MouseMoved {
        position: &'a PhysicalPosition<f64>,
    },
    MouseWheel {
        delta: &'a MouseScrollDelta,
    },
    Draw,
}

pub struct Window {
    event_loop: EventLoop<()>,
    pub window: window::Window,
}

impl Window {
    #[cfg(target_arch = "wasm32")]
    fn init_canvas(window: &winit::window::Window) {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(800, 800));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    pub fn new() -> Self {
        // TODO: Add size
        let event_loop = EventLoop::new();
        let window = window::WindowBuilder::new()
            .with_title("MJOLNIR")
            .build(&event_loop)
            .unwrap();

        #[cfg(target_arch = "wasm32")]
        Window::init_canvas(&window);

        Self { event_loop, window }
    }

    pub fn run(self, mut callback: impl 'static + FnMut(WindowEvents)) {
        self.event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == self.window.id() => {
                    // Handle window events (like resizing, or key inputs)
                    // This is stuff from `winit` -- see their docs for more info
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state,
                                    virtual_keycode: Some(keycode),
                                    ..
                                },
                            ..
                        } => callback(WindowEvents::Keyboard {
                            state: *state,
                            virtual_keycode: keycode,
                        }),
                        WindowEvent::Resized(physical_size) => callback(WindowEvents::Resized {
                            width: physical_size.width,
                            height: physical_size.height,
                        }),
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &&mut so w have to dereference it twice
                            callback(WindowEvents::Resized {
                                width: new_inner_size.width,
                                height: new_inner_size.height,
                            })
                        }
                        WindowEvent::MouseInput { state, button, .. } => {
                            callback(WindowEvents::MouseInput { state, button })
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            callback(WindowEvents::MouseMoved { position })
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            callback(WindowEvents::MouseWheel { delta })
                        }
                        _ => {}
                    }
                }
                Event::RedrawRequested(window_id) if window_id == self.window.id() => {
                    callback(WindowEvents::Draw);
                }
                Event::RedrawEventsCleared => {
                    // RedrawRequested will only trigger once, unless we manually
                    // request it.
                    self.window.request_redraw();
                }

                _ => {}
            }
        });
    }
}
