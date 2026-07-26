use ethers::types::U256;
use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    /// The `mine` compute pipeline, compiled ONCE at construction and reused by
    /// every `dispatch_mine` batch. Compiling the full secp256k1+keccak MSL per
    /// batch was the GPU path's dominant cost (it made the GPU slower than CPU).
    mine_pipeline: ComputePipelineState,
    /// The `mine_create2` pipeline (keccak-only), compiled once and reused by
    /// every `dispatch_create2` batch. CREATE2 salt mining needs no secp256k1.
    create2_pipeline: ComputePipelineState,
    /// The `mine_create3` pipeline (keccak-only, two keccaks/candidate),
    /// compiled once and reused by every `dispatch_create3` batch. CREATE3 salt
    /// mining needs no secp256k1.
    create3_pipeline: ComputePipelineState,
    /// The `mine_createx` pipeline (keccak-only, three keccaks/candidate: the
    /// CreateX salt guard, then the CREATE3 proxy + final). Compiled once,
    /// reused by every `dispatch_createx` batch.
    createx_pipeline: ComputePipelineState,
}

/// CreateX (pcaversaccio) canonical cross-chain address — the CREATE2 deployer
/// of the CREATE3 proxy, identical on every chain that ran the Nick's-method
/// pre-signed deploy. Source of truth:
/// <https://github.com/pcaversaccio/createx> README pin
/// `0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed`. Override at runtime with
/// `CREATEX_FACTORY` / `CREATEX_ADDRESS` (see `miner::createx`).
pub const CREATEX_ADDRESS: [u8; 20] = [
    0xba, 0x5E, 0xd0, 0x99, 0x63, 0x3D, 0x3B, 0x31, 0x3e, 0x4D, 0x5F, 0x7b, 0xdc, 0x13, 0x05, 0xd3,
    0xc2, 0x8b, 0xa5, 0xEd,
];

/// Standard CREATE3 proxy init-code hash.
/// CreateX `proxyChildBytecode` = `hex"67_36_3d_3d_37_36_3d_34_f0_3d_52_60_08_60_18_f3"`
/// (<https://github.com/pcaversaccio/createx> `src/CreateX.sol`); Solmate /
/// 0xSequence share the same 16-byte proxy. `keccak256` of that bytecode:
/// `0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f`.
pub const STANDARD_CREATE3_PROXY_HASH: [u8; 32] = [
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54, 0x2f,
    0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49, 0x7c, 0x1f,
];

/// A verified-candidate mining hit: a private key whose derived Ethereum
/// address has `zeros` leading zero nibbles (counted by the same semantics as
/// `src/main.rs`'s CPU tool). Callers MUST re-verify via
/// `MetalContext::verify_hit` before trusting `address` -- see `dispatch_mine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub privkey: [u8; 32],
    pub address: [u8; 20],
    pub zeros: u8,
}

/// A CREATE2 vanity-salt hit: a salt whose resulting contract address (from a
/// fixed deployer + init-code hash) has `zeros` leading zero nibbles. Callers
/// MUST re-verify via `MetalContext::verify_create2` before trusting `address`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Create2Hit {
    pub salt: [u8; 32],
    pub address: [u8; 20],
    pub zeros: u8,
}

/// A CREATE3 vanity-salt hit: a salt whose resulting contract address (from a
/// fixed CREATE3 factory + its constant proxy init-code hash) has `zeros`
/// leading zero nibbles. The address is bytecode-independent — it depends only
/// on (factory, salt). Callers MUST re-verify via
/// `MetalContext::verify_create3` before trusting `address`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Create3Hit {
    pub salt: [u8; 32],
    pub address: [u8; 20],
    pub zeros: u8,
}

/// Must match `HIT_STRIDE` in `kernels/miner.metal`.
const MINE_HIT_STRIDE: usize = 64;
/// Must match `MAX_HITS` in `kernels/miner.metal`.
const MINE_MAX_HITS: usize = 1024;

impl MetalContext {
    pub fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let mine_pipeline = Self::build_mine_pipeline(&device);
        let create2_pipeline = Self::build_create2_pipeline(&device);
        let create3_pipeline = Self::build_create3_pipeline(&device);
        let createx_pipeline = Self::build_createx_pipeline(&device);
        Self {
            device,
            queue,
            mine_pipeline,
            create2_pipeline,
            create3_pipeline,
            createx_pipeline,
        }
    }

    /// Compile the CreateX MSL (keccak + createx) and build the `mine_createx`
    /// pipeline. Called once from `new()`, reused by every `dispatch_createx`.
    fn build_createx_pipeline(device: &Device) -> ComputePipelineState {
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let createx_src = include_str!("../../kernels/createx.metal");
        let src = format!("{keccak_src}\n{createx_src}");
        let lib = device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("createx kernel compile failed");
        let func = lib
            .get_function("mine_createx", None)
            .expect("mine_createx entry not found");
        device
            .new_compute_pipeline_state_with_function(&func)
            .expect("createx pipeline")
    }

    /// Compile the CREATE3 MSL (keccak + create3, no secp256k1) and build the
    /// `mine_create3` pipeline. Called once from `new()`, reused by every
    /// `dispatch_create3` call.
    fn build_create3_pipeline(device: &Device) -> ComputePipelineState {
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let create3_src = include_str!("../../kernels/create3.metal");
        let src = format!("{keccak_src}\n{create3_src}");
        let lib = device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("create3 kernel compile failed");
        let func = lib
            .get_function("mine_create3", None)
            .expect("mine_create3 entry not found");
        device
            .new_compute_pipeline_state_with_function(&func)
            .expect("create3 pipeline")
    }

    /// Compile the CREATE2 MSL (keccak + create2, no secp256k1) and build the
    /// `mine_create2` pipeline. Called once from `new()`, reused by every
    /// `dispatch_create2` call.
    fn build_create2_pipeline(device: &Device) -> ComputePipelineState {
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let create2_src = include_str!("../../kernels/create2.metal");
        let src = format!("{keccak_src}\n{create2_src}");
        let lib = device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("create2 kernel compile failed");
        let func = lib
            .get_function("mine_create2", None)
            .expect("mine_create2 entry not found");
        device
            .new_compute_pipeline_state_with_function(&func)
            .expect("create2 pipeline")
    }

    /// Assemble the concatenated miner MSL (keccak+field+ec+miner) and build the
    /// `mine` compute pipeline. Called once from `new()`; the result is cached in
    /// `mine_pipeline` and reused by every `dispatch_mine` call, so the hot
    /// mining loop never recompiles the kernel.
    fn build_mine_pipeline(device: &Device) -> ComputePipelineState {
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");
        let lib = device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("miner kernel compile failed");
        let func = lib
            .get_function("mine", None)
            .expect("mine entry not found");
        device
            .new_compute_pipeline_state_with_function(&func)
            .expect("mine pipeline")
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
        let func = lib
            .get_function("field_test", None)
            .expect("entry not found");
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

    /// Test-only: apply `scalar_add_small` (the incremental-walk's
    /// `(base + it) mod n` arithmetic) on the GPU for each `(base, it)` pair
    /// via `kernels/miner.metal`'s `scalar_add_mod_n_test`, returning the
    /// result as a `U256`. `base` must already be reduced mod n (< n).
    pub fn run_scalar_add_mod_n(&self, bases: &[U256], its: &[u32]) -> Vec<U256> {
        assert_eq!(bases.len(), its.len(), "bases/its length mismatch");
        let n = bases.len();
        let mut base_limbs = vec![0u32; n * 8];
        for (i, b) in bases.iter().enumerate() {
            base_limbs[i * 8..i * 8 + 8].copy_from_slice(&u256_to_limbs(*b));
        }

        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib
            .get_function("scalar_add_mod_n_test", None)
            .expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let base_buf = self.device.new_buffer_with_data(
            base_limbs.as_ptr() as *const std::ffi::c_void,
            (base_limbs.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let its_buf = self.device.new_buffer_with_data(
            its.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(its) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_len = n * 8;
        let out_buf = self.device.new_buffer(
            (out_len * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&base_buf), 0);
        enc.set_buffer(1, Some(&its_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const u32;
        let limbs = unsafe { std::slice::from_raw_parts(ptr, out_len) };
        (0..n)
            .map(|i| {
                let mut l = [0u32; 8];
                l.copy_from_slice(&limbs[i * 8..i * 8 + 8]);
                limbs_to_u256(&l)
            })
            .collect()
    }

    /// Test-only: run the incremental Jacobian walk (Approach B's core loop)
    /// from `iters` explicit 256-bit `bases` (not keccak-derived), via
    /// `kernels/miner.metal`'s `mine_incremental_raw_test`. Returns, per base,
    /// `iters` `(privkey, address)` pairs for `(base+0)*G .. (base+iters-1)*G`.
    /// Lets tests target exact scalar values (e.g. the n-wrap boundary).
    pub fn run_mine_incremental_raw(
        &self,
        bases: &[[u8; 32]],
        iters: u32,
    ) -> Vec<Vec<([u8; 32], [u8; 20])>> {
        let n = bases.len();
        let mut in_limbs = vec![0u32; n * 8];
        for (i, k) in bases.iter().enumerate() {
            for limb in 0..8 {
                let hi = 28 - limb * 4;
                in_limbs[i * 8 + limb] =
                    u32::from_be_bytes([k[hi], k[hi + 1], k[hi + 2], k[hi + 3]]);
            }
        }

        // NOTE (corrected post-Task-2 review): `miner.metal`'s pre-existing
        // `pubkey_to_address` (called by our new kernel) and other kernels
        // call `keccak256`, so the whole file needs `keccak.metal` present
        // in the concatenation even though our entry point doesn't call it
        // directly -- MSL requires every symbol referenced anywhere in the
        // translation unit to resolve, not just in the dispatched kernel's
        // call graph. Use the same 4-file concatenation `dispatch_mine` uses.
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib
            .get_function("mine_incremental_raw_test", None)
            .expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let in_buf = self.device.new_buffer_with_data(
            in_limbs.as_ptr() as *const std::ffi::c_void,
            (in_limbs.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let iters_arr = [iters];
        let iters_buf = self.device.new_buffer_with_data(
            iters_arr.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let priv_len = n * (iters as usize) * 32;
        let addr_len = n * (iters as usize) * 20;
        let out_priv_buf = self
            .device
            .new_buffer(priv_len as u64, MTLResourceOptions::StorageModeShared);
        let out_addr_buf = self
            .device
            .new_buffer(addr_len as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&iters_buf), 0);
        enc.set_buffer(2, Some(&out_priv_buf), 0);
        enc.set_buffer(3, Some(&out_addr_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let priv_ptr = out_priv_buf.contents() as *const u8;
        let priv_bytes = unsafe { std::slice::from_raw_parts(priv_ptr, priv_len) };
        let addr_ptr = out_addr_buf.contents() as *const u8;
        let addr_bytes = unsafe { std::slice::from_raw_parts(addr_ptr, addr_len) };

        (0..n)
            .map(|i| {
                (0..iters as usize)
                    .map(|it| {
                        let mut pk = [0u8; 32];
                        pk.copy_from_slice(
                            &priv_bytes[(i * iters as usize + it) * 32..(i * iters as usize + it) * 32 + 32],
                        );
                        let mut addr = [0u8; 20];
                        addr.copy_from_slice(
                            &addr_bytes[(i * iters as usize + it) * 20..(i * iters as usize + it) * 20 + 20],
                        );
                        (pk, addr)
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute the affine secp256k1 public key `k*G` for each 32-byte
    /// big-endian private key on the GPU via `kernels/ec.metal` (concatenated
    /// after `kernels/field.metal`, mirroring `run_field`'s source assembly).
    /// Returns one 64-byte `x‖y` per key, each coordinate 32 big-endian bytes,
    /// matching k256's uncompressed encoding (`0x04 ‖ x ‖ y`) minus the prefix.
    pub fn run_scalarmul(&self, keys: &[[u8; 32]]) -> Vec<[u8; 64]> {
        let n = keys.len();
        // Marshal each big-endian key into 8 little-endian u32 limbs.
        let mut in_limbs = vec![0u32; n * 8];
        for (i, k) in keys.iter().enumerate() {
            for limb in 0..8 {
                let hi = 28 - limb * 4; // limb 0 = least-significant word
                in_limbs[i * 8 + limb] =
                    u32::from_be_bytes([k[hi], k[hi + 1], k[hi + 2], k[hi + 3]]);
            }
        }

        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let src = format!("{field_src}\n{ec_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib.get_function("ec_test", None).expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let in_buf = self.device.new_buffer_with_data(
            in_limbs.as_ptr() as *const std::ffi::c_void,
            (in_limbs.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_len = n * 16; // 8 x-limbs + 8 y-limbs per key
        let out_buf = self.device.new_buffer(
            (out_len * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const u32;
        let limbs = unsafe { std::slice::from_raw_parts(ptr, out_len) };
        (0..n)
            .map(|i| {
                let mut out = [0u8; 64];
                // x limbs -> bytes [0..32] BE, y limbs -> bytes [32..64] BE.
                for limb in 0..8 {
                    let pos = 28 - limb * 4;
                    out[pos..pos + 4].copy_from_slice(&limbs[i * 16 + limb].to_be_bytes());
                    out[32 + pos..32 + pos + 4]
                        .copy_from_slice(&limbs[i * 16 + 8 + limb].to_be_bytes());
                }
                out
            })
            .collect()
    }

    /// Derive `(privkey, address)` on the GPU for each `(seed, counter)` pair
    /// via `kernels/miner.metal` (concatenated after keccak/field/ec, mirroring
    /// `run_scalarmul`'s source assembly): `privkey = keccak256(seed‖counter_le8)`
    /// (scalar-range-guarded), `address = keccak256(x_be‖y_be)[12..32]` of the
    /// resulting `privkey*G` point. A guard miss (privkey == 0 or >= the
    /// secp256k1 order) yields an all-zero address for that entry.
    pub fn derive_address_gpu(
        &self,
        seeds: &[[u8; 32]],
        counters: &[u64],
    ) -> Vec<([u8; 32], [u8; 20])> {
        assert_eq!(
            seeds.len(),
            counters.len(),
            "seeds/counters length mismatch"
        );
        let n = seeds.len();

        let mut seed_bytes = vec![0u8; n * 32];
        for (i, s) in seeds.iter().enumerate() {
            seed_bytes[i * 32..i * 32 + 32].copy_from_slice(s);
        }

        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib
            .get_function("derive_test", None)
            .expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let seeds_buf = self.device.new_buffer_with_data(
            seed_bytes.as_ptr() as *const std::ffi::c_void,
            seed_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counters_buf = self.device.new_buffer_with_data(
            counters.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(counters) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_priv_buf = self
            .device
            .new_buffer((n * 32) as u64, MTLResourceOptions::StorageModeShared);
        let out_addr_buf = self
            .device
            .new_buffer((n * 20) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&seeds_buf), 0);
        enc.set_buffer(1, Some(&counters_buf), 0);
        enc.set_buffer(2, Some(&out_priv_buf), 0);
        enc.set_buffer(3, Some(&out_addr_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let priv_ptr = out_priv_buf.contents() as *const u8;
        let priv_bytes = unsafe { std::slice::from_raw_parts(priv_ptr, n * 32) };
        let addr_ptr = out_addr_buf.contents() as *const u8;
        let addr_bytes = unsafe { std::slice::from_raw_parts(addr_ptr, n * 20) };

        (0..n)
            .map(|i| {
                let mut pk = [0u8; 32];
                pk.copy_from_slice(&priv_bytes[i * 32..i * 32 + 32]);
                let mut addr = [0u8; 20];
                addr.copy_from_slice(&addr_bytes[i * 20..i * 20 + 20]);
                (pk, addr)
            })
            .collect()
    }

    /// Search `iters` candidate keys per thread (one thread per
    /// `(seed, base_counter)` pair, counters `base_counters[i]..base_counters[i]+iters`)
    /// via `kernels/miner.metal`'s `mine` kernel (Approach A: every candidate
    /// is an independent keccak-derived scalar, full scalar-mult each),
    /// returning every derived address with `>= threshold` leading-zero
    /// nibbles (bounded to at most 1024 hits per call -- extras in a
    /// saturated batch are silently dropped on the GPU side; callers on a
    /// tight loop should keep `threshold` high enough that a batch rarely
    /// saturates).
    ///
    /// SECURITY: every returned `Hit` is a GPU-derived candidate only. Callers
    /// MUST re-verify each one with `verify_hit` before treating `address` as
    /// trustworthy (that host-side gate is what the CLI's per-hit hard-abort
    /// enforces).
    pub fn dispatch_mine(
        &self,
        seeds: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Hit> {
        self.dispatch_mine_kernel("mine", seeds, base_counters, iters, threshold)
    }

    /// Search `iters` candidate keys per thread via `kernels/miner.metal`'s
    /// `mine_incremental` kernel (Approach B: one keccak-derived base scalar
    /// per thread per batch, then `iters` cheap Jacobian point additions
    /// instead of `iters` full scalar multiplications -- see
    /// `docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` for the
    /// cost model and the security-model discussion of why this is weaker
    /// than, but not unsafe compared to, `dispatch_mine`'s Approach A).
    /// Same wire format, same buffer layout, same 1024-hit cap as `mine`.
    ///
    /// SECURITY: same requirement as `dispatch_mine` -- callers MUST
    /// re-verify every returned `Hit` with `verify_hit` before trusting it.
    pub fn dispatch_mine_incremental(
        &self,
        seeds: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Hit> {
        self.dispatch_mine_kernel("mine_incremental", seeds, base_counters, iters, threshold)
    }

    /// Shared dispatch body for `dispatch_mine`/`dispatch_mine_incremental`:
    /// both kernels have an identical buffer signature (seeds, base_counters,
    /// iters, threshold, hit_count, hits), differing only in per-candidate
    /// math, so only the compiled entry-point name changes.
    fn dispatch_mine_kernel(
        &self,
        entry: &str,
        seeds: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Hit> {
        assert_eq!(
            seeds.len(),
            base_counters.len(),
            "seeds/base_counters length mismatch"
        );
        let n = seeds.len();

        let mut seed_bytes = vec![0u8; n * 32];
        for (i, s) in seeds.iter().enumerate() {
            seed_bytes[i * 32..i * 32 + 32].copy_from_slice(s);
        }

        // Reuse the pre-compiled `mine` pipeline (built once in `new()`) — a hot
        // mining loop no longer recompiles the full MSL on every batch.
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib.get_function(entry, None).expect("entry not found");
        let _pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let seeds_buf = self.device.new_buffer_with_data(
            seed_bytes.as_ptr() as *const std::ffi::c_void,
            seed_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counters_buf = self.device.new_buffer_with_data(
            base_counters.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(base_counters) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let iters_buf = self.device.new_buffer_with_data(
            &iters as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let threshold_buf = self.device.new_buffer_with_data(
            &threshold as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // MTLBuffer contents are undefined until written -- zero the atomic
        // counter explicitly rather than relying on incidental zero pages.
        unsafe {
            std::ptr::write_bytes(
                hit_count_buf.contents() as *mut u8,
                0,
                std::mem::size_of::<u32>(),
            );
        }
        let hits_bytes = MINE_MAX_HITS * MINE_HIT_STRIDE;
        let hits_buf = self
            .device
            .new_buffer(hits_bytes as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.mine_pipeline);
        enc.set_buffer(0, Some(&seeds_buf), 0);
        enc.set_buffer(1, Some(&counters_buf), 0);
        enc.set_buffer(2, Some(&iters_buf), 0);
        enc.set_buffer(3, Some(&threshold_buf), 0);
        enc.set_buffer(4, Some(&hit_count_buf), 0);
        enc.set_buffer(5, Some(&hits_buf), 0);
        // Threadgroup sizing per Apple's compute guidance ("Calculating
        // threadgroup and grid sizes"): a 1-thread threadgroup underuses the
        // GPU's SIMD width (~32 lanes idle out of every 32). Use dispatch_threads
        // (non-uniform threadgroups; macOS 10.13+/Apple Silicon) with the widest
        // threadgroup the pipeline permits, capped at the grid size. Per Apple,
        // dispatch_threads needs no in-kernel bounds check — gid stays in [0, n).
        let tg = self
            .mine_pipeline
            .max_total_threads_per_threadgroup()
            .min(n as u64)
            .max(1);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let found = unsafe { *(hit_count_buf.contents() as *const u32) } as usize;
        let count = found.min(MINE_MAX_HITS);

        let ptr = hits_buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, hits_bytes) };

        (0..count)
            .map(|i| {
                let rec = i * MINE_HIT_STRIDE;
                let mut privkey = [0u8; 32];
                privkey.copy_from_slice(&bytes[rec..rec + 32]);
                let mut address = [0u8; 20];
                address.copy_from_slice(&bytes[rec + 32..rec + 52]);
                let zeros = bytes[rec + 52];
                Hit {
                    privkey,
                    address,
                    zeros,
                }
            })
            .collect()
    }

    /// Host-side re-derivation gate: recompute the Ethereum address from
    /// `privkey` via `k256`/`ethers::utils::secret_key_to_address` and check
    /// it matches `address`. Returns `false` (not a panic) for a
    /// non-canonical `privkey` (e.g. zero or >= the secp256k1 order), so
    /// scalar-range-guard misses cleanly fail the check.
    pub fn verify_hit(&self, privkey: [u8; 32], address: [u8; 20]) -> bool {
        use ethers::core::k256::ecdsa::SigningKey;
        use ethers::utils::secret_key_to_address;
        match SigningKey::from_bytes((&privkey).into()) {
            Ok(sk) => secret_key_to_address(&sk).as_bytes() == address,
            Err(_) => false,
        }
    }

    /// Search `iters` salts per thread via `kernels/create2.metal`'s
    /// `mine_create2` kernel: for a fixed `deployer` (20 bytes) and
    /// `initcodehash` (32 bytes), return every CREATE2 address with `>= threshold`
    /// leading-zero nibbles (bounded to 1024 hits/call, like `dispatch_mine`).
    /// Each thread's salt is `base_salts[gid]` with its low 8 bytes replaced by
    /// `base_counters[gid] + it`.
    ///
    /// SECURITY: every returned `Create2Hit` is a GPU candidate; callers MUST
    /// re-verify with `verify_create2` before trusting `address`.
    pub fn dispatch_create2(
        &self,
        deployer: &[u8; 20],
        initcodehash: &[u8; 32],
        base_salts: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Create2Hit> {
        assert_eq!(
            base_salts.len(),
            base_counters.len(),
            "base_salts/base_counters length mismatch"
        );
        let n = base_salts.len();

        let mut salt_bytes = vec![0u8; n * 32];
        for (i, s) in base_salts.iter().enumerate() {
            salt_bytes[i * 32..i * 32 + 32].copy_from_slice(s);
        }

        let deployer_buf = self.device.new_buffer_with_data(
            deployer.as_ptr() as *const std::ffi::c_void,
            20,
            MTLResourceOptions::StorageModeShared,
        );
        let ich_buf = self.device.new_buffer_with_data(
            initcodehash.as_ptr() as *const std::ffi::c_void,
            32,
            MTLResourceOptions::StorageModeShared,
        );
        let salts_buf = self.device.new_buffer_with_data(
            salt_bytes.as_ptr() as *const std::ffi::c_void,
            salt_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counters_buf = self.device.new_buffer_with_data(
            base_counters.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(base_counters) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let iters_buf = self.device.new_buffer_with_data(
            &iters as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let threshold_buf = self.device.new_buffer_with_data(
            &threshold as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::write_bytes(
                hit_count_buf.contents() as *mut u8,
                0,
                std::mem::size_of::<u32>(),
            );
        }
        let hits_bytes = MINE_MAX_HITS * MINE_HIT_STRIDE;
        let hits_buf = self
            .device
            .new_buffer(hits_bytes as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.create2_pipeline);
        enc.set_buffer(0, Some(&deployer_buf), 0);
        enc.set_buffer(1, Some(&ich_buf), 0);
        enc.set_buffer(2, Some(&salts_buf), 0);
        enc.set_buffer(3, Some(&counters_buf), 0);
        enc.set_buffer(4, Some(&iters_buf), 0);
        enc.set_buffer(5, Some(&threshold_buf), 0);
        enc.set_buffer(6, Some(&hit_count_buf), 0);
        enc.set_buffer(7, Some(&hits_buf), 0);
        // Same threadgroup-occupancy sizing as dispatch_mine.
        let tg = self
            .create2_pipeline
            .max_total_threads_per_threadgroup()
            .min(n as u64)
            .max(1);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let found = unsafe { *(hit_count_buf.contents() as *const u32) } as usize;
        let count = found.min(MINE_MAX_HITS);
        let ptr = hits_buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, hits_bytes) };

        (0..count)
            .map(|i| {
                let rec = i * MINE_HIT_STRIDE;
                let mut salt = [0u8; 32];
                salt.copy_from_slice(&bytes[rec..rec + 32]);
                let mut address = [0u8; 20];
                address.copy_from_slice(&bytes[rec + 32..rec + 52]);
                let zeros = bytes[rec + 52];
                Create2Hit {
                    salt,
                    address,
                    zeros,
                }
            })
            .collect()
    }

    /// Host-side re-derivation gate for a CREATE2 hit: recompute
    /// `keccak256(0xff ‖ deployer ‖ salt ‖ initcodehash)[12..32]` and compare it
    /// to `address`. Returns `false` on any mismatch (a GPU bug must never emit a
    /// bad salt).
    pub fn verify_create2(
        &self,
        deployer: &[u8; 20],
        initcodehash: &[u8; 32],
        salt: &[u8; 32],
        address: &[u8; 20],
    ) -> bool {
        use ethers::utils::keccak256;
        let mut preimage = Vec::with_capacity(85);
        preimage.push(0xff);
        preimage.extend_from_slice(deployer);
        preimage.extend_from_slice(salt);
        preimage.extend_from_slice(initcodehash);
        &keccak256(&preimage)[12..32] == address
    }

    /// Dispatch one CREATE3 keccak-only batch: `mine_create3` scans
    /// `iters` salts per thread for `factory` + `proxyhash`, reporting salts
    /// whose FINAL (bytecode-independent) address has `>= threshold` leading
    /// zero nibbles. Layout mirrors `dispatch_create2`; buffer(1) carries the
    /// proxy init-code hash instead of the contract init-code hash. Every hit
    /// MUST be re-verified via `verify_create3` before it is trusted.
    pub fn dispatch_create3(
        &self,
        factory: &[u8; 20],
        proxyhash: &[u8; 32],
        base_salts: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Create3Hit> {
        assert_eq!(
            base_salts.len(),
            base_counters.len(),
            "base_salts/base_counters length mismatch"
        );
        let n = base_salts.len();

        let mut salt_bytes = vec![0u8; n * 32];
        for (i, s) in base_salts.iter().enumerate() {
            salt_bytes[i * 32..i * 32 + 32].copy_from_slice(s);
        }

        let factory_buf = self.device.new_buffer_with_data(
            factory.as_ptr() as *const std::ffi::c_void,
            20,
            MTLResourceOptions::StorageModeShared,
        );
        let ph_buf = self.device.new_buffer_with_data(
            proxyhash.as_ptr() as *const std::ffi::c_void,
            32,
            MTLResourceOptions::StorageModeShared,
        );
        let salts_buf = self.device.new_buffer_with_data(
            salt_bytes.as_ptr() as *const std::ffi::c_void,
            salt_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counters_buf = self.device.new_buffer_with_data(
            base_counters.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(base_counters) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let iters_buf = self.device.new_buffer_with_data(
            &iters as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let threshold_buf = self.device.new_buffer_with_data(
            &threshold as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::write_bytes(
                hit_count_buf.contents() as *mut u8,
                0,
                std::mem::size_of::<u32>(),
            );
        }
        let hits_bytes = MINE_MAX_HITS * MINE_HIT_STRIDE;
        let hits_buf = self
            .device
            .new_buffer(hits_bytes as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.create3_pipeline);
        enc.set_buffer(0, Some(&factory_buf), 0);
        enc.set_buffer(1, Some(&ph_buf), 0);
        enc.set_buffer(2, Some(&salts_buf), 0);
        enc.set_buffer(3, Some(&counters_buf), 0);
        enc.set_buffer(4, Some(&iters_buf), 0);
        enc.set_buffer(5, Some(&threshold_buf), 0);
        enc.set_buffer(6, Some(&hit_count_buf), 0);
        enc.set_buffer(7, Some(&hits_buf), 0);
        let tg = self
            .create3_pipeline
            .max_total_threads_per_threadgroup()
            .min(n as u64)
            .max(1);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let found = unsafe { *(hit_count_buf.contents() as *const u32) } as usize;
        let count = found.min(MINE_MAX_HITS);
        let ptr = hits_buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, hits_bytes) };

        (0..count)
            .map(|i| {
                let rec = i * MINE_HIT_STRIDE;
                let mut salt = [0u8; 32];
                salt.copy_from_slice(&bytes[rec..rec + 32]);
                let mut address = [0u8; 20];
                address.copy_from_slice(&bytes[rec + 32..rec + 52]);
                let zeros = bytes[rec + 52];
                Create3Hit {
                    salt,
                    address,
                    zeros,
                }
            })
            .collect()
    }

    /// Host-side re-derivation of a CREATE3 address, used to verify every GPU
    /// hit before it is trusted (a mismatch indicates a kernel/host divergence
    /// or a wrong `proxyhash`, and the caller hard-aborts):
    ///   proxy = keccak256(0xff ‖ factory ‖ salt ‖ proxyhash)[12:]
    ///   addr  = keccak256(0xd6 ‖ 0x94 ‖ proxy ‖ 0x01)[12:]
    pub fn verify_create3(
        &self,
        factory: &[u8; 20],
        proxyhash: &[u8; 32],
        salt: &[u8; 32],
        address: &[u8; 20],
    ) -> bool {
        use ethers::utils::keccak256;
        // Step 1: proxy = CREATE2(factory, salt, proxyhash)
        let mut pre = Vec::with_capacity(85);
        pre.push(0xff);
        pre.extend_from_slice(factory);
        pre.extend_from_slice(salt);
        pre.extend_from_slice(proxyhash);
        let proxy = keccak256(&pre);
        // Step 2: addr = RLP([proxy, nonce=1]) -> keccak -> low 20 bytes
        let mut rlp = Vec::with_capacity(23);
        rlp.push(0xd6);
        rlp.push(0x94);
        rlp.extend_from_slice(&proxy[12..32]);
        rlp.push(0x01);
        &keccak256(&rlp)[12..32] == address
    }

    /// Dispatch one CreateX permissionless-CREATE3 batch: `mine_createx` scans
    /// `iters` salts per thread, guarding each (`guardedSalt = keccak256(salt)`)
    /// and computing the CreateX CREATE3 address, reporting salts whose FINAL
    /// address has `>= threshold` leading zero nibbles. `createx`/`proxyhash`
    /// are the canonical constants (`CREATEX_ADDRESS` / `STANDARD_CREATE3_PROXY_HASH`).
    /// The emitted salt is the ORIGINAL (un-guarded) salt to pass to
    /// `CreateX.deployCreate3`. Every hit MUST be re-verified via `verify_createx`.
    pub fn dispatch_createx(
        &self,
        createx: &[u8; 20],
        proxyhash: &[u8; 32],
        base_salts: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Create3Hit> {
        assert_eq!(
            base_salts.len(),
            base_counters.len(),
            "base_salts/base_counters length mismatch"
        );
        let n = base_salts.len();

        let mut salt_bytes = vec![0u8; n * 32];
        for (i, s) in base_salts.iter().enumerate() {
            salt_bytes[i * 32..i * 32 + 32].copy_from_slice(s);
        }

        let createx_buf = self.device.new_buffer_with_data(
            createx.as_ptr() as *const std::ffi::c_void,
            20,
            MTLResourceOptions::StorageModeShared,
        );
        let ph_buf = self.device.new_buffer_with_data(
            proxyhash.as_ptr() as *const std::ffi::c_void,
            32,
            MTLResourceOptions::StorageModeShared,
        );
        let salts_buf = self.device.new_buffer_with_data(
            salt_bytes.as_ptr() as *const std::ffi::c_void,
            salt_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counters_buf = self.device.new_buffer_with_data(
            base_counters.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(base_counters) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let iters_buf = self.device.new_buffer_with_data(
            &iters as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let threshold_buf = self.device.new_buffer_with_data(
            &threshold as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::write_bytes(
                hit_count_buf.contents() as *mut u8,
                0,
                std::mem::size_of::<u32>(),
            );
        }
        let hits_bytes = MINE_MAX_HITS * MINE_HIT_STRIDE;
        let hits_buf = self
            .device
            .new_buffer(hits_bytes as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.createx_pipeline);
        enc.set_buffer(0, Some(&createx_buf), 0);
        enc.set_buffer(1, Some(&ph_buf), 0);
        enc.set_buffer(2, Some(&salts_buf), 0);
        enc.set_buffer(3, Some(&counters_buf), 0);
        enc.set_buffer(4, Some(&iters_buf), 0);
        enc.set_buffer(5, Some(&threshold_buf), 0);
        enc.set_buffer(6, Some(&hit_count_buf), 0);
        enc.set_buffer(7, Some(&hits_buf), 0);
        let tg = self
            .createx_pipeline
            .max_total_threads_per_threadgroup()
            .min(n as u64)
            .max(1);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let found = unsafe { *(hit_count_buf.contents() as *const u32) } as usize;
        let count = found.min(MINE_MAX_HITS);
        let ptr = hits_buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, hits_bytes) };

        (0..count)
            .map(|i| {
                let rec = i * MINE_HIT_STRIDE;
                let mut salt = [0u8; 32];
                salt.copy_from_slice(&bytes[rec..rec + 32]);
                let mut address = [0u8; 20];
                address.copy_from_slice(&bytes[rec + 32..rec + 52]);
                let zeros = bytes[rec + 52];
                Create3Hit {
                    salt,
                    address,
                    zeros,
                }
            })
            .collect()
    }

    /// Host-side re-derivation of a CreateX permissionless CREATE3 address:
    ///   guarded = keccak256(salt)
    ///   proxy   = keccak256(0xff ‖ createx ‖ guarded ‖ proxyhash)[12:]
    ///   addr    = keccak256(0xd6 ‖ 0x94 ‖ proxy ‖ 0x01)[12:]
    pub fn verify_createx(
        &self,
        createx: &[u8; 20],
        proxyhash: &[u8; 32],
        salt: &[u8; 32],
        address: &[u8; 20],
    ) -> bool {
        use ethers::utils::keccak256;
        let guarded = keccak256(salt); // permissionless branch: keccak256(abi.encode(salt))
        let mut pre = Vec::with_capacity(85);
        pre.push(0xff);
        pre.extend_from_slice(createx);
        pre.extend_from_slice(&guarded);
        pre.extend_from_slice(proxyhash);
        let proxy = keccak256(&pre);
        let mut rlp = Vec::with_capacity(23);
        rlp.push(0xd6);
        rlp.push(0x94);
        rlp.extend_from_slice(&proxy[12..32]);
        rlp.push(0x01);
        &keccak256(&rlp)[12..32] == address
    }

    /// Compile `src`, bind `inputs`/`lens`/an output buffer, dispatch the
    /// `keccak_test` kernel over `n` threads (one per input), and return the
    /// resulting 32-byte digests.
    fn dispatch_keccak(&self, src: &str, inputs: &[u8], lens: &[u32], n: usize) -> Vec<[u8; 32]> {
        let lib = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib
            .get_function("keccak_test", None)
            .expect("entry not found");
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
            std::mem::size_of_val(lens) as u64,
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
            .as_chunks::<32>()
            .0
            .iter()
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
