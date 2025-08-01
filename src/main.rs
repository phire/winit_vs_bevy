use clap::{Parser, ValueEnum};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "winit_vs_bevy")]
#[command(about = "A comparison between custom wgpu+winit and Bevy rendering")]
struct Args {
    /// Renderer to use
    #[arg(short, long, default_value = "raw")]
    renderer: Renderer,
}

#[derive(Clone, ValueEnum)]
enum Renderer {
    /// Raw wgpu+winit implementation
    Raw,
    /// Bevy 0.16 implementation
    Bevy,
}

fn main() {
    let args = Args::parse();

    match args.renderer {
        Renderer::Raw => {
            env_logger::init();
            run_raw_renderer();
        }
        Renderer::Bevy => run_bevy_renderer(),
    }
}

fn run_raw_renderer() {
    use winit::{
        application::ApplicationHandler,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
        window::{Window, WindowId},
    };

    println!("Running raw wgpu+winit renderer...");

    struct App {
        window: Option<Arc<Window>>,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
        surface: Option<wgpu::Surface<'static>>,
        surface_config: Option<wgpu::SurfaceConfiguration>,
        // FPS tracking
        frame_count: u32,
        last_fps_log: Instant,
    }

    impl Default for App {
        fn default() -> Self {
            Self {
                window: None,
                device: None,
                queue: None,
                surface: None,
                surface_config: None,
                frame_count: 0,
                last_fps_log: Instant::now(),
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let window_attributes = Window::default_attributes()
                .with_title("Raw WGPU Blank Window")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

            let window = Arc::new(
                event_loop
                    .create_window(window_attributes)
                    .expect("Failed to create window")
            );

            // Initialize wgpu
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            });

            let surface = instance
                .create_surface(Arc::clone(&window))
                .expect("Failed to create surface");

            // Request adapter
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }))
            .expect("Failed to find adapter");

            // Request device and queue
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor::default(),
            ))
            .expect("Failed to create device");

            // Configure the surface
            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps
                .formats
                .iter()
                .find(|f| f.is_srgb())
                .copied()
                .unwrap_or(surface_caps.formats[0]);

            println!(
                "Using surface format: {:?}",
                surface_format
            );


            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };

            surface.configure(&device, &config);

            self.window = Some(window);
            self.device = Some(device);
            self.queue = Some(queue);
            self.surface = Some(surface);
            self.surface_config = Some(config);
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.render();
                }
                WindowEvent::Resized(physical_size) => {
                    if let (Some(surface), Some(device), Some(config)) = (
                        &self.surface,
                        &self.device,
                        &mut self.surface_config,
                    ) {
                        if physical_size.width > 0 && physical_size.height > 0 {
                            config.width = physical_size.width;
                            config.height = physical_size.height;
                            surface.configure(device, config);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    impl App {
        fn render(&mut self) {
            if let (Some(surface), Some(device), Some(queue)) =
                (&self.surface, &self.device, &self.queue)
            {
                let output = surface
                    .get_current_texture()
                    .expect("Failed to get surface texture");

                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

                // Simply starting a render pass triggers a clear operation.
                {
                    let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                }

                queue.submit(std::iter::once(encoder.finish()));
                output.present();

                // Not a great FPS counter, just here as a sanity check to make sure we are vsynced

                // Update FPS tracking
                self.frame_count += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(self.last_fps_log);

                // Log FPS every 5 seconds
                if elapsed.as_secs() >= 5 {
                    let fps = self.frame_count as f64 / elapsed.as_secs_f64();
                    println!("Raw Renderer FPS: {:.2}", fps);
                    self.frame_count = 0;
                    self.last_fps_log = now;
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();

    event_loop
        .run_app(&mut app)
        .expect("Failed to run event loop");
}

fn run_bevy_renderer() {
    use bevy::prelude::*;
    use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};

    println!("Running Bevy 0.16 renderer...");

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Bevy Blank Window".into(),
                    resolution: (800.0, 600.0).into(),
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin{
                debug: false,
                wait_duration: std::time::Duration::from_secs(5),
                filter: None,
            }
        ))
        .insert_resource(ClearColor(Color::srgb(0.1, 0.2, 0.3)))
        .run();
}
