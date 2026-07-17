use metal::{CommandQueue, CompileOptions, Device, MTLResourceOptions, MTLSize};

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
}

impl MetalContext {
    pub fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        Self { device, queue }
    }

    /// Compile `src`, dispatch `entry` over `tgroups*tperg` threads writing a
    /// `u32` output buffer of `out_len` elements, and return its contents.
    pub fn run_u32_kernel(
        &self,
        src: &str,
        entry: &str,
        out_len: usize,
        tgroups: u64,
        tperg: u64,
    ) -> Vec<u32> {
        let lib = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib.get_function(entry, None).expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let bytes = (out_len * std::mem::size_of::<u32>()) as u64;
        let out_buf = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&out_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(tgroups, 1, 1), MTLSize::new(tperg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec()
    }

    /// Hash each of `inputs` (each <= 64 bytes) with the Ethereum Keccak-256
    /// MSL kernel in `kernels/keccak.metal`, returning one 32-byte digest per
    /// input, in order.
    pub fn run_keccak_fixed64(&self, inputs: &[Vec<u8>]) -> Vec<[u8; 32]> {
        const STRIDE: usize = 64;
        let n = inputs.len();
        let mut flat = vec![0u8; n * STRIDE];
        let mut lens = vec![0u32; n];
        for (i, inp) in inputs.iter().enumerate() {
            assert!(inp.len() <= STRIDE);
            flat[i * STRIDE..i * STRIDE + inp.len()].copy_from_slice(inp);
            lens[i] = inp.len() as u32;
        }
        let src = include_str!("../../kernels/keccak.metal");
        self.dispatch_keccak(src, &flat, &lens, n)
    }

    /// Compile `src`, bind `inputs`/`lens`/an output buffer, dispatch the
    /// `keccak_test` kernel over `n` threads (one per input), and return the
    /// resulting 32-byte digests.
    fn dispatch_keccak(&self, src: &str, inputs: &[u8], lens: &[u32], n: usize) -> Vec<[u8; 32]> {
        let lib = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib.get_function("keccak_test", None).expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let in_buf = self.device.new_buffer_with_data(
            inputs.as_ptr() as *const std::ffi::c_void,
            inputs.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let lens_buf = self.device.new_buffer_with_data(
            lens.as_ptr() as *const std::ffi::c_void,
            (lens.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_bytes = (n * 32) as u64;
        let out_buf = self
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&lens_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, n * 32) };
        bytes
            .chunks_exact(32)
            .map(|c| {
                let mut out = [0u8; 32];
                out.copy_from_slice(c);
                out
            })
            .collect()
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}
