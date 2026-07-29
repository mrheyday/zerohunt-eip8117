# CreateX integration (nullforge)

Nullforge’s `--createx` mode mines **permissionless CREATE3** vanity salts for
[CreateX](https://github.com/pcaversaccio/createx) (`pcaversaccio/createx`).

## Upstream

| Item              | Value                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------- |
| Repo              | https://github.com/pcaversaccio/createx                                                                  |
| Contract          | `src/CreateX.sol`                                                                                        |
| Interface         | `src/ICreateX.sol`                                                                                       |
| Canonical address | `0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed` (Nick’s method, every chain that ran the pre-signed deploy) |
| License           | AGPL-3.0-only                                                                                            |

## What we implement

Permissionless branch only (`SenderBytes.Random` in `CreateX._guard`):

```
guardedSalt = keccak256(salt)
proxy       = CREATE2(createx, guardedSalt, proxyChildBytecode)
addr        = RLP_CREATE(proxy, nonce=1)
```

`proxyChildBytecode` = `0x67363d3d37363d34f03d5260086018f3`  
`keccak256(proxyChildBytecode)` = `0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f`  
(same as Solmate / 0xSequence CREATE3 proxy)

The miner emits the **original** salt for `deployCreate3(salt, initCode)`; the
guard is re-applied on-chain.

## Factory override

Default factory is the canonical address. For a project-local CreateX redeploy
(different CREATE address, same CREATE3 math):

```sh
export CREATEX_FACTORY=0xc0d239e75615F9b5763a9918493C03437370bDbB
# or CREATEX_ADDRESS=...
nullforge-gpu 8 --createx
```

## Files

| Path                    | Role                                                     |
| ----------------------- | -------------------------------------------------------- |
| `kernels/createx.metal` | GPU: guard + CREATE3                                     |
| `src/miner/createx.rs`  | Host driver + factory resolve                            |
| `src/gpu/mod.rs`        | `CREATEX_ADDRESS`, `STANDARD_CREATE3_PROXY_HASH`, verify |

Do **not** invent a second CREATE3 formula — re-diff against
`CreateX.sol` if upstream changes `proxyChildBytecode` or `_guard`.
