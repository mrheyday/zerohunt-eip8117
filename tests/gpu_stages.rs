use zerohunt::gpu::MetalContext;

#[test]
fn echo_kernel_doubles_thread_id() {
    let ctx = MetalContext::new();
    let src = include_str!("../kernels/echo.metal");
    let out = ctx.run_u32_kernel(src, "echo", 256, 4, 64);
    assert_eq!(out.len(), 256);
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, (i as u32) * 2, "thread {i} wrong");
    }
}
