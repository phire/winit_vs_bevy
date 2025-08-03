
use clap::{Parser, ValueEnum};
use eframe::egui_wgpu::WgpuConfiguration;
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
    /// Direct Metal implementation (macOS only)
    #[cfg(target_os = "macos")]
    Metal,
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
        #[cfg(target_os = "macos")]
        Renderer::Metal => run_metal_renderer(args.no_vsync),
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

#[cfg(target_os = "macos")]
fn run_metal_renderer(no_vsync: bool) {
    // Roughly based on the example from https://docs.rs/objc2-metal/0.3.1/objc2_metal/index.html
    // Adjusted to match xcode's game template
    use core::cell::OnceCell;
    use std::cell::RefCell;

    use objc2::{define_class, msg_send, rc::Retained, runtime::ProtocolObject, DefinedClass, MainThreadOnly, MainThreadMarker};

    use objc2_app_kit::*;
    use objc2_foundation::*;
    use objc2_metal::*;
    use objc2_metal_kit::*;

    use block2::RcBlock;
    use dispatch2::{DispatchRetained, DispatchSemaphore, DispatchTime};

    println!("Running direct Metal renderer...");

    struct Ivars {
        window: OnceCell<Retained<NSWindow>>,
        command_queue: OnceCell<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
        semaphore: OnceCell<DispatchRetained<DispatchSemaphore>>,
        fps_tracker: RefCell<FpsTracker>,
    }

    define_class!(
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[ivars = Ivars]

        struct Delegate;

        unsafe impl NSObjectProtocol for Delegate {}

        unsafe impl NSApplicationDelegate for Delegate {
            #[unsafe(method(applicationDidFinishLaunching:))]
            #[allow(non_snake_case)]
            unsafe fn applicationDidFinishLaunching(&self, notification: &NSNotification) {

                let mtm = MainThreadMarker::from(self);
                let app = unsafe { notification.object() }
                    .unwrap()
                    .downcast::<NSApplication>()
                    .expect("Failed to get NSApplication");

                // Activate the application
                unsafe { app.activate(); }

                // create the window
                let window = {
                let content_rect = NSRect::new(NSPoint::new(0., 0.), NSSize::new(800., 600.));
                let style = NSWindowStyleMask::Closable
                    | NSWindowStyleMask::Resizable
                    | NSWindowStyleMask::Titled;
                let backing_store_type = NSBackingStoreType::Buffered;
                let flag = false;
                unsafe {
                        NSWindow::initWithContentRect_styleMask_backing_defer(
                            mtm.alloc(),
                            content_rect,
                            style,
                            backing_store_type,
                            flag,
                        )
                    }
                };

                // get the default device
                let device =
                    MTLCreateSystemDefaultDevice().expect("failed to get metal device");

                // create the command queue
                let command_queue = device
                    .newCommandQueue()
                    .expect("Failed to create a command queue.");

                // create the metal view
                let mtk_view = {
                    let frame_rect = window.frame();
                    unsafe { MTKView::initWithFrame_device(mtm.alloc(), frame_rect, Some(&device)) }
                };

                // attach this delegate to the MTKView
                let object = ProtocolObject::from_ref(self);
                unsafe { mtk_view.setDelegate(Some(object)) };

                // Request 120 fps, otherwise MTKView defaults to 60 fps
                unsafe { mtk_view.setPreferredFramesPerSecond(120); }

                // Set a custom clear color
                unsafe { mtk_view.setClearColor(MTLClearColor { red: 0.1, green: 0.2, blue: 0.2, alpha: 1.0 }); }

                // finish setting up the window
                window.setContentView(Some(&mtk_view));
                window.center();
                window.setTitle(ns_string!("Native Metal Blank Window"));
                window.makeKeyAndOrderFront(None);

                // Maximum of 3 frames in flight
                let semaphore = DispatchSemaphore::new(3);

                self.ivars().command_queue.set(command_queue).unwrap();
                self.ivars().semaphore.set(semaphore).unwrap();
                self.ivars().window.set(window).unwrap();
            }

            #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
            fn should_terminate_after_last_window_closed(&self, _: &NSApplication) -> bool { true }
        }

        unsafe impl MTKViewDelegate for Delegate {
            #[unsafe(method(drawInMTKView:))]
            #[allow(non_snake_case)]
            unsafe fn drawInMTKView(&self, mtk_view: &MTKView) {
                let semaphore = self.ivars().semaphore.get().unwrap();
                let command_queue = self.ivars().command_queue.get().unwrap();

                semaphore.wait(DispatchTime::FOREVER);

                // Update FPS tracking
                self.ivars().fps_tracker.borrow_mut().update();

                let Some(pass_descriptor) = (unsafe { mtk_view.currentRenderPassDescriptor() }) else {
                    println!("No render pass available");
                    return;
                };
                let current_drawable = unsafe { mtk_view.currentDrawable() }.unwrap();
                let command_buffer = command_queue.commandBuffer().unwrap();

                // Register a completion handler to signal the semaphore when done
                let semaphore = semaphore.clone();
                unsafe { command_buffer.addCompletedHandler(RcBlock::into_raw(RcBlock::new(move |_| {
                    semaphore.signal();
                }))) };

                // Start render pass (clears screen)
                let Some(encoder) = command_buffer.renderCommandEncoderWithDescriptor(&pass_descriptor) else {
                    return;
                };

                // finish render pass
                encoder.endEncoding();

                // Present the drawable and commit
                command_buffer.presentDrawable(ProtocolObject::from_ref(&*current_drawable));
                command_buffer.commit();
            }

            #[unsafe(method(mtkView:drawableSizeWillChange:))]
            #[allow(non_snake_case)]
            unsafe fn mtkView_drawableSizeWillChange(&self, _view: &MTKView, _size: NSSize) {
                // don't need to do anything, mtkView will handle resizing automatically
            }
        }
    );

    impl Delegate {
        fn new(mtm: MainThreadMarker) -> Retained<Self> {
            let this = Self::alloc(mtm);

            let fps_tracker = RefCell::new(FpsTracker::new("Metal Renderer"));
            let this = this.set_ivars(Ivars {
                window: Default::default(),
                command_queue: Default::default(),
                semaphore: Default::default(),
                fps_tracker,
            });
            unsafe { msg_send![super(this), init] }
        }
    }

    // Create the app
    let mtm = MainThreadMarker::new().unwrap();
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    // attach our delegate
    let delegate = Delegate::new(mtm);
    let object = ProtocolObject::from_ref(&*delegate);
    app.setDelegate(Some(object));

    // Run the app
    app.run();
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
