
use clap::{Parser, ValueEnum};
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

// Unified options struct for all renderers
#[derive(Clone)]
struct RendererOptions {
    pub no_vsync: bool,
    pub triangle: bool,
}

impl From<Args> for RendererOptions {
    fn from(args: Args) -> Self {
        Self {
            no_vsync: args.no_vsync,
            triangle: args.triangle,
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
    /// Render triangles instead of blank screen
    #[arg(long)]
    triangle: bool,
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
    let renderer = args.renderer.clone();
    let options = RendererOptions::from(args);

    match renderer {
        Renderer::Raw => {
            env_logger::init();
            run_raw_renderer(&options);
        }
        Renderer::Bevy => run_bevy_renderer(&options),
        Renderer::Eframe => run_eframe_renderer(false, &options),
        Renderer::EframeWgpu => run_eframe_renderer(true, &options),
        #[cfg(target_os = "macos")]
        Renderer::Metal => run_metal_renderer(&options),
    }
}

fn run_raw_renderer(options: &RendererOptions) {
    use winit::{
        application::ApplicationHandler,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
        window::{Window, WindowId},
    };

    // Unified structs for 3D rendering
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct Vertex {
        position: [f32; 3],
        normal: [f32; 3],
    }

    unsafe impl bytemuck::Pod for Vertex {}
    unsafe impl bytemuck::Zeroable for Vertex {}

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct Uniforms {
        view_proj: [[f32; 4]; 4],
        light_position: [f32; 3],
        _padding: f32,
        light_color: [f32; 3],
        _padding2: f32,
    }

    unsafe impl bytemuck::Pod for Uniforms {}
    unsafe impl bytemuck::Zeroable for Uniforms {}

    println!("Running raw wgpu+winit renderer...");

    struct App {
        window: Option<Arc<Window>>,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
        surface: Option<wgpu::Surface<'static>>,
        surface_config: Option<wgpu::SurfaceConfiguration>,
        render_pipeline: Option<wgpu::RenderPipeline>,
        vertex_buffer: Option<wgpu::Buffer>,
        uniform_buffer: Option<wgpu::Buffer>,
        bind_group: Option<wgpu::BindGroup>,
        fps_tracker: FpsTracker,
        options: RendererOptions,
    }

    impl App {
        fn new(options: RendererOptions) -> Self {
            Self {
                window: None,
                device: None,
                queue: None,
                surface: None,
                surface_config: None,
                render_pipeline: None,
                vertex_buffer: None,
                uniform_buffer: None,
                bind_group: None,
                fps_tracker: FpsTracker::new("Raw Renderer"),
                options,
            }
        }
    }

    impl Default for App {
        fn default() -> Self {
            Self::new(RendererOptions {
                no_vsync: false,
                triangle: false,
            })
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
            let present_mode = if self.options.no_vsync {
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

            // Create triangle pipeline and vertex buffer if requested
            let (render_pipeline, vertex_buffer, uniform_buffer, bind_group) = if self.options.triangle {
                impl Vertex {
                    fn desc() -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    offset: 0,
                                    shader_location: 0,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                                wgpu::VertexAttribute {
                                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                                    shader_location: 1,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                            ],
                        }
                    }
                }

                // Triangle vertices matching Bevy's triangle
                let vertices = [
                    Vertex { position: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] },
                    Vertex { position: [-1.0, -1.0, 0.0], normal: [0.0, 0.0, 1.0] },
                    Vertex { position: [1.0, -1.0, 0.0], normal: [0.0, 0.0, 1.0] },
                ];

                let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Vertex Buffer"),
                    size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));

                // Create view and projection matrices using cgmath (matching Bevy's camera setup)
                let eye = cgmath::Point3::new(0.0, 0.0, 9.0);
                let target = cgmath::Point3::new(0.0, 0.0, 0.0);
                let up = cgmath::Vector3::unit_y();

                let view = cgmath::Matrix4::look_at_rh(eye, target, up);
                let projection = cgmath::perspective(
                    cgmath::Deg(45.0),
                    size.width as f32 / size.height as f32,
                    0.1,
                    100.0
                );

                let view_proj = projection * view;

                let uniforms = Uniforms {
                    view_proj: view_proj.into(),
                    light_position: [4.0, 8.0, 4.0], // Matching Bevy's light position
                    _padding: 0.0,
                    light_color: [1.0, 1.0, 1.0],
                    _padding2: 0.0,
                };

                let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Uniform Buffer"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

                // Create bind group layout
                let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("uniform_bind_group_layout"),
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    }],
                    label: Some("uniform_bind_group"),
                });

                // Enhanced vertex shader with 3D transformations and lighting
                let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Vertex Shader"),
                    source: wgpu::ShaderSource::Wgsl(r#"
                        struct Uniforms {
                            view_proj: mat4x4<f32>,
                            light_position: vec3<f32>,
                            light_color: vec3<f32>,
                        }
                        @group(0) @binding(0)
                        var<uniform> uniforms: Uniforms;

                        struct VertexInput {
                            @location(0) position: vec3<f32>,
                            @location(1) normal: vec3<f32>,
                        }

                        struct VertexOutput {
                            @builtin(position) clip_position: vec4<f32>,
                            @location(0) world_position: vec3<f32>,
                            @location(1) normal: vec3<f32>,
                        }

                        @vertex
                        fn vs_main(model: VertexInput) -> VertexOutput {
                            var out: VertexOutput;
                            out.world_position = model.position;
                            out.normal = model.normal;
                            out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
                            return out;
                        }
                    "#.into()),
                });

                // Enhanced fragment shader with Phong lighting
                let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Fragment Shader"),
                    source: wgpu::ShaderSource::Wgsl(r#"
                        struct Uniforms {
                            view_proj: mat4x4<f32>,
                            light_position: vec3<f32>,
                            light_color: vec3<f32>,
                        }
                        @group(0) @binding(0)
                        var<uniform> uniforms: Uniforms;

                        struct VertexOutput {
                            @builtin(position) clip_position: vec4<f32>,
                            @location(0) world_position: vec3<f32>,
                            @location(1) normal: vec3<f32>,
                        }

                        @fragment
                        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                            let material_color = vec3<f32>(1.0, 1.0, 1.0);
                            let ambient = vec3<f32>(0.1, 0.1, 0.1);

                            let light_dir = normalize(uniforms.light_position - in.world_position);
                            let normal = normalize(in.normal);

                            let diffuse = max(dot(normal, light_dir), 0.0) * uniforms.light_color;

                            let final_color = material_color * (ambient + diffuse);
                            return vec4<f32>(final_color, 1.0);
                        }
                    "#.into()),
                });

                let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Render Pipeline Layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    })),
                    vertex: wgpu::VertexState {
                        module: &vs_module,
                        entry_point: Some("vs_main"),
                        buffers: &[Vertex::desc()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs_module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

                (Some(render_pipeline), Some(vertex_buffer), Some(uniform_buffer), Some(bind_group))
            } else {
                (None, None, None, None)
            };

            self.window = Some(window);
            self.device = Some(device);
            self.queue = Some(queue);
            self.surface = Some(surface);
            self.surface_config = Some(config);
            self.render_pipeline = render_pipeline;
            self.vertex_buffer = vertex_buffer;
            self.uniform_buffer = uniform_buffer;
            self.bind_group = bind_group;
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
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

                    // Draw triangle if enabled
                    if let (Some(pipeline), Some(vertex_buffer), Some(bind_group), Some(uniform_buffer)) =
                        (&self.render_pipeline, &self.vertex_buffer, &self.bind_group, &self.uniform_buffer) {

                        // Update uniforms every frame (even if static) to match Bevy's behavior
                        let size = self.surface_config.as_ref().unwrap();

                        #[repr(C)]
                        #[derive(Copy, Clone, Debug)]
                        struct Uniforms {
                            view_proj: [[f32; 4]; 4],
                            light_position: [f32; 3],
                            _padding: f32,
                            light_color: [f32; 3],
                            _padding2: f32,
                        }

                        unsafe impl bytemuck::Pod for Uniforms {}
                        unsafe impl bytemuck::Zeroable for Uniforms {}

                        // Recreate matrices every frame (even though they're static)
                        let eye = cgmath::Point3::new(0.0, 0.0, 9.0);
                        let target = cgmath::Point3::new(0.0, 0.0, 0.0);
                        let up = cgmath::Vector3::unit_y();

                        let view = cgmath::Matrix4::look_at_rh(eye, target, up);
                        let projection = cgmath::perspective(
                            cgmath::Deg(45.0),
                            size.width as f32 / size.height as f32,
                            0.1,
                            100.0
                        );

                        let view_proj = projection * view;

                        let uniforms = Uniforms {
                            view_proj: view_proj.into(),
                            light_position: [4.0, 8.0, 4.0],
                            _padding: 0.0,
                            light_color: [1.0, 1.0, 1.0],
                            _padding2: 0.0,
                        };

                        // Upload uniforms every frame
                        queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(0, bind_group, &[]);
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        render_pass.draw(0..3, 0..1);
                    }
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

    let mut app = App::new(options.clone());

    event_loop
        .run_app(&mut app)
        .expect("Failed to run event loop");
}

fn run_bevy_renderer(options: &RendererOptions) {
    use bevy::prelude::*;
    use bevy::render::RenderApp;
    use bevy::window::PresentMode;

    println!("Running Bevy 0.16 renderer...");

    #[derive(Resource)]
    struct BevyFpsTracker(FpsTracker);

    fn fps_tracking_system(mut fps_tracker: ResMut<BevyFpsTracker>) {
        fps_tracker.0.update();
    }

    fn setup(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 0.0, 9.0).looking_at(Vec3::ZERO, Vec3::Y),

        ));

        commands.spawn((
            PointLight {
                shadows_enabled: true,
                ..default()
            },
            Transform::from_xyz(4.0, 8.0, 4.0),
        ));

        for i in 0..1 {
            let triangle = meshes.add(Triangle3d::new(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
            ));
            commands.spawn((
                Mesh3d(triangle),
                MeshMaterial3d(materials.add( Color::WHITE)),
                Transform::from_xyz(0.0, -0.1 * i as f32, -0.1 * i as f32),
            ));
        }


    }

    let mut app = App::new();

    app
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Blank Window".into(),
                resolution: (800.0, 600.0).into(),
                present_mode: if options.no_vsync { PresentMode::AutoNoVsync } else { PresentMode::AutoVsync },
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.1, 0.2, 0.3)));

    if options.triangle {
        app.add_systems(Startup, setup);
    }

    let render_app = (&mut app).get_sub_app_mut(RenderApp).expect("RenderApp not found");

    render_app
        .insert_resource(BevyFpsTracker(FpsTracker::new("Bevy Renderer")))
        .add_systems(ExtractSchedule, fps_tracking_system);


    app.run();
}

#[cfg(target_os = "macos")]
fn run_metal_renderer(options: &RendererOptions) {
    // Check for unsupported options
    if options.triangle {
        eprintln!("Error: Triangle rendering is not yet implemented for the Metal renderer");
        eprintln!("Try using --renderer raw or --renderer bevy instead");
        std::process::exit(1);
    }

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
        options: RendererOptions,
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
        fn new(mtm: MainThreadMarker, options: RendererOptions) -> Retained<Self> {
            let this = Self::alloc(mtm);

            let fps_tracker = RefCell::new(FpsTracker::new("Metal Renderer"));
            let this = this.set_ivars(Ivars {
                window: Default::default(),
                command_queue: Default::default(),
                semaphore: Default::default(),
                fps_tracker,
                options,
            });
            unsafe { msg_send![super(this), init] }
        }
    }

    // Create the app
    let mtm = MainThreadMarker::new().unwrap();
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    // attach our delegate
    let delegate = Delegate::new(mtm, options.clone());
    let object = ProtocolObject::from_ref(&*delegate);
    app.setDelegate(Some(object));

    // Run the app
    app.run();
}


fn run_eframe_renderer(use_wgpu: bool, options: &RendererOptions) {
    use eframe::egui;
    use eframe::egui_wgpu::WgpuConfiguration;

    // Check for unsupported options
    if options.triangle {
        eprintln!("Error: Triangle rendering is not supported by the eframe/egui renderer");
        eprintln!("Try using --renderer raw or --renderer bevy instead");
        std::process::exit(1);
    }

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
            [0.1, 0.2, 0.3, 1.0] // set clear color
        }

        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            // FPS tracking
            self.fps_tracker.update();

            // Request repaint
            ctx.request_repaint();
        }
    }

    let window_title = if use_wgpu { "Eframe WGPU Blank Window" } else { "Eframe Blank Window" };

    let eframe_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title(window_title),
        renderer: if use_wgpu { eframe::Renderer::Wgpu } else { eframe::Renderer::Glow },
        vsync: !options.no_vsync, // Only used by Glow.
        wgpu_options: WgpuConfiguration {
            present_mode: if options.no_vsync {
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
        eframe_options,
        Box::new(move |_cc| Ok(Box::new(EframeApp::new(use_wgpu)))),
    ).expect("Failed to run eframe app");
}
