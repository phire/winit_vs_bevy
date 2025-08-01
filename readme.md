Simple App to test the raw overhead of bevy-render vs raw wgpu+winit

Bevy has a [issue](https://github.com/bevyengine/bevy/issues/10261) on MacOS where a simple 3d scene seems to result in well-over 50% CPU usage.


Even simply created a window with absolutely nothing in it seems to use more CPU usage than it should,
especially compared to creating a window in winit. 

But, [I'm not sure](https://github.com/bevyengine/bevy/issues/10261#issuecomment-3142080407) this is
actually a bug in Bevy (or any of it's dependencies or macos). 

The "bug" is that "CPU usage" is an absolutely useless metric these days, especially on very energy efficient CPU's like Apple's M1.

The reason why a raw winit window uses less CPU than a raw bevy-render window, is that the raw winit
window isn't rendering anything, while bevy-render defaults to vsynced rendering of a blank frame.

This example is designed to be a more fair comparison, using wgpu to actually render a blank frame.

To test winit+wgpu, launch with: `cargo run -r -- -r raw`. To test bevy, launch with: `cargo run -r -- -r bevy`.

## Results

In my quick testing on my M1-Max:
(window focused, no mouse movement, wait for usage in Activity Monitor to stabilize)

 * Raw winit+wgpu: 15.5% CPU
 * Bevy: 18.3% CPU

IMO, this is close enough that they might as well be identical. 
If anything, the fact Bevy is only using 3% more CPU time is surprising because it's doing way more work under the hood (extracting entities from the world, building scene graphs) before even submitting a blank frame, and then somehow managing to trigger two 128KB resource allocations per frame.

The raw gpu+winit has zero allocations and is hardcoded to always directly submit a blank frame.

## Notes

This example explicitly avoids the bevy `multi_threaded` feature. Partly because it's just pointless overhead for anything so simple, partly because it does use quite a bit of "CPU time" to synchronize across all the threads, but mostly because it's a huge pain to profile.

When the multi_threaded feature is enabled, the CPU usage on my machine is more like 32%. If you are
concerned about "CPU usage", then consider disabling it. Though, I doubt this extra CPU usage translates into a real world performance difference, especially with a more complex scene.

## Key takeaway:

CPU usage simply is not a useful metric. It's bad enough on x86 machines, but absolutely terrible on Apple Silicon.

