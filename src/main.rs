
use clap::{Parser, ValueEnum};
use eframe::egui_wgpu::{self, WgpuConfiguration};
use std::sync::Arc;
use std::time::Instant;

// Unified FPS tracking
struct FpsTracker {
    frame_count: u32,
    last_fps_log: Instant,
    renderer_name: String,
}

impl FpsTracker {
    fn new(renderer_name: &str) -> Self {
        Self {
            frame_count: 0,
            last_fps_log: Instant::now(),
            renderer_name: renderer_name.to_string(),
        }
    }

    fn update(&mut self) {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_log);

        // Log FPS every 5 seconds
        if elapsed.as_secs() >= 5 {
            let fps = self.frame_count as f64 / elapsed.as_secs_f64();
            println!("{} FPS: {:.2}", self.renderer_name, fps);
            self.frame_count = 0;
            self.last_fps_log = now;
        }

    }
}

#[derive(Parser)]
#[command(name = "winit_vs_bevy")]
#[command(about = "A comparison between custom wgpu+winit and Bevy rendering")]
struct Args {
    /// Renderer to use
    #[arg(short, long, default_value = "raw")]
    renderer: Renderer,
    /// Disable vsync (vertical synchronization)
    #[arg(long)]
    no_vsync: bool,
}

#[derive(Clone, ValueEnum)]
enum Renderer {
    /// Raw wgpu+winit implementation
    Raw,
    /// Bevy 0.16 implementation
    Bevy,
    /// eframe/egui implementation (default backend)
    Eframe,
    /// eframe/egui implementation with wgpu backend
    EframeWgpu,
}

fn main() {
    let args = Args::parse();

    match args.renderer {
        Renderer::Raw => {
            env_logger::init();
            run_raw_renderer(args.no_vsync);
        }
        Renderer::Bevy => run_bevy_renderer(args.no_vsync),
        Renderer::Eframe => run_eframe_renderer(false, args.no_vsync),
        Renderer::EframeWgpu => run_eframe_renderer(true, args.no_vsync),
    }
}

fn run_raw_renderer(no_vsync: bool) {
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
        fps_tracker: FpsTracker,
        no_vsync: bool,
    }

    impl App {
        fn new(no_vsync: bool) -> Self {
            Self {
                window: None,
                device: None,
                queue: None,
                surface: None,
                surface_config: None,
                fps_tracker: FpsTracker::new("Raw Renderer"),
                no_vsync,
            }
        }
    }

    impl Default for App {
        fn default() -> Self {
            Self::new(false)
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

            println!(
                "Using adapter: {} ({:?})",
                adapter.get_info().name,
                adapter.get_info().device_type
            );

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
            let present_mode = if self.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            };
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode,
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

                // Update FPS tracking
                self.fps_tracker.update();

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(no_vsync);

    event_loop
        .run_app(&mut app)
        .expect("Failed to run event loop");
}

fn run_bevy_renderer(no_vsync: bool) {
    use bevy::prelude::*;

    println!("Running Bevy 0.16 renderer...");

    #[derive(Resource)]
    struct BevyFpsTracker(FpsTracker);

    fn fps_tracking_system(mut fps_tracker: ResMut<BevyFpsTracker>) {
        fps_tracker.0.update();
    }

    let present_mode = if no_vsync {
        bevy::window::PresentMode::Immediate
    } else {
        bevy::window::PresentMode::AutoVsync
    };

    App::new()
        .insert_resource(BevyFpsTracker(FpsTracker::new("Bevy Renderer")))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Blank Window".into(),
                resolution: (800.0, 600.0).into(),
                present_mode,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Update, fps_tracking_system)
        .insert_resource(ClearColor(Color::srgb(0.1, 0.2, 0.3)))
        .run();
}

fn run_eframe_renderer(use_wgpu: bool, no_vsync: bool) {
    use eframe::egui;

    let backend_name = if use_wgpu { "with wgpu backend" } else { "" };
    println!("Running eframe/egui renderer {}...", backend_name);

    struct EframeApp {
        fps_tracker: FpsTracker,
    }

    impl EframeApp {
        fn new(use_wgpu: bool) -> Self {
            let backend = if use_wgpu { "WGPU" } else { "Glow" };
            Self {
                fps_tracker: FpsTracker::new(&format!("Eframe {}", backend)),
            }
        }
    }

    impl eframe::App for EframeApp {
        fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
            [0.1, 0.3, 0.2, 1.0] // set clear color
        }

        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            // FPS tracking
            self.fps_tracker.update();

            // Request repaint
            ctx.request_repaint();
        }
    }

    let window_title = if use_wgpu { "Eframe WGPU Blank Window" } else { "Eframe Blank Window" };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title(window_title),
        renderer: if use_wgpu { eframe::Renderer::Wgpu } else { eframe::Renderer::Glow },
        vsync: !no_vsync, // Only used by Glow.
        wgpu_options: WgpuConfiguration {
            present_mode: if no_vsync {
                eframe::wgpu::PresentMode::AutoNoVsync
            } else {
                eframe::wgpu::PresentMode::AutoVsync
            },
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        window_title,
        options,
        Box::new(move |_cc| Ok(Box::new(EframeApp::new(use_wgpu)))),
    ).expect("Failed to run eframe app");
}
