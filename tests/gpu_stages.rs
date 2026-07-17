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

#[test]
fn keccak_matches_host_reference() {
    use ethers::utils::keccak256;
    let ctx = MetalContext::new();
    let inputs: Vec<Vec<u8>> = vec![
        vec![],
        b"abc".to_vec(),
        (0u8..64).collect(),
    ];
    let gpu = ctx.run_keccak_fixed64(&inputs);
    for (i, inp) in inputs.iter().enumerate() {
        assert_eq!(gpu[i], keccak256(inp), "keccak mismatch on input {i}");
    }
}
