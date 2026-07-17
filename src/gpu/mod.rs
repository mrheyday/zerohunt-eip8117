use ethers::types::U256;
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

    /// Apply one secp256k1 field operation (`op`: 0=add, 1=sub, 2=mul, 3=inv)
    /// to the operand pair `(a, b)` on the GPU via `kernels/field.metal`, and
    /// return the result as a reduced `U256` in `[0, p)`. `a`/`b` must already
    /// be reduced mod p. Operands marshal to 8 little-endian u32 limbs.
    pub fn run_field(&self, a: U256, b: U256, op: u32) -> U256 {
        let a_limbs = u256_to_limbs(a);
        let b_limbs = u256_to_limbs(b);
        let src = include_str!("../../kernels/field.metal");

        let lib = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib.get_function("field_test", None).expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let limb_bytes = (8 * std::mem::size_of::<u32>()) as u64;
        let a_buf = self.device.new_buffer_with_data(
            a_limbs.as_ptr() as *const std::ffi::c_void,
            limb_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = self.device.new_buffer_with_data(
            b_limbs.as_ptr() as *const std::ffi::c_void,
            limb_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let op_arr = [op];
        let op_buf = self.device.new_buffer_with_data(
            op_arr.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self
            .device
            .new_buffer(limb_bytes, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_buffer(3, Some(&op_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const u32;
        let limbs = unsafe { std::slice::from_raw_parts(ptr, 8) };
        let mut out = [0u32; 8];
        out.copy_from_slice(limbs);
        limbs_to_u256(&out)
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

/// Marshal a `U256` into 8 little-endian u32 limbs (limb 0 = least significant).
fn u256_to_limbs(x: U256) -> [u32; 8] {
    let mut bytes = [0u8; 32];
    x.to_little_endian(&mut bytes);
    let mut limbs = [0u32; 8];
    for (i, limb) in limbs.iter_mut().enumerate() {
        *limb = u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    limbs
}

/// Reassemble a `U256` from 8 little-endian u32 limbs.
fn limbs_to_u256(limbs: &[u32; 8]) -> U256 {
    let mut bytes = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&limb.to_le_bytes());
    }
    U256::from_little_endian(&bytes)
}
