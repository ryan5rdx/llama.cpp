# Testing tensor parallelism over Apple RDMA

How to exercise `-sm tensor` across two Apple silicon Macs joined by a Thunderbolt cable,
on branch `apple-rdma-tp-fast-sync`.

Nothing in the fast-sync or batching path has ever run on real hardware. The Metal
primitives underneath are tested locally; the RPC restructuring, the zero-copy send and
the frame-size negotiation are not. Expect to find bugs, and read
[What is unproven](#what-is-unproven) before trusting a number.

Throughout, `A` and `B` are the two nodes, `$TB_A` and `$TB_B` their Thunderbolt
addresses.

---

## 1. One-time setup

### RDMA

On **both** nodes:

```bash
rdma_ctl status          # expect: disabled
```

Enabling is a boot-policy change and must be done from macOS Recovery:

```bash
rdma_ctl enable
```

Reboot, then confirm `rdma_ctl status` reports `enabled` on both. Needs macOS 26.2 or
later. See [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

### Link

Thunderbolt 5 cable directly between the two Macs. Give the Thunderbolt Bridge interface
a static address on each, e.g. `10.77.0.1` on A and `10.77.0.2` on B. Check both
directions before going further:

```bash
ssh A "ping -c3 10.77.0.2"
ssh B "ping -c3 10.77.0.1"
```

### Provider capabilities

Two things the zero-copy send needs are unverified on this hardware. Answer them once,
before spending time on anything else:

```bash
# copy tools/rpc/rdma_caps.c (or the local copy) to a node and run it - no peer needed
clang -O2 rdma_caps.c -o rdma_caps -lrdma && ./rdma_caps
```

Look for:

- `ibv_reg_mr over vm_allocate memory` - if this fails, `send_from` can never engage and
  the zero-copy send is dead code. Everything else still works.
- `max_send_sge` - the zero-copy send needs 3. If the provider grants fewer, it silently
  falls back to the bounce copy.

### Build

Identical commit on both nodes:

```bash
for h in A B; do ssh $h "cd llama.cpp && git checkout apple-rdma-tp-fast-sync && \
  cmake -S . -B build-tp -DGGML_METAL=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build-tp -j --target ggml-rpc-server llama-bench llama-cli"; done
```

Confirm the configure output on both says:

```
RDMA transport enabled (Apple RDMA-over-Thunderbolt, UC)
```

A model is only needed on the **client** node. The servers receive tensors over the wire;
`-c` caches them locally so later runs skip the upload.

---

## 2. Addressing, which is where this goes wrong

Apple Thunderbolt RDMA is point to point. Each cable is its own subnet, so a node's
address depends on which cable you are asking about. Three consequences:

**The client must run on A or B, not a third machine.** `ggml_backend_rpc_comm_init`
derives rank 0's address by parsing the client's own `--rpc` string and ships it to
rank 1. From a third machine that address is on the client-to-A cable, which B has no
route to. It either fails to connect or, worse, silently reaches A over Ethernet and the
peer link runs on TCP while the client links run on RDMA - a healthy-looking, meaningless
benchmark.

**`--rpc` must name A by its Thunderbolt address even though the client is on A.**
Writing `127.0.0.1:50052` makes the client ship `127.0.0.1` to B, and B dials its own
loopback. Same trap with A's Ethernet address: B would reach A over Ethernet and the peer
link would be TCP.

**`-H` decides where rank 0 listens for its peer.** `comm_init` binds the comm listener
to whatever `-H` was given. Bind to `127.0.0.1` and the peer link never leaves loopback.

Order in `--rpc` matters: the first entry becomes rank 0 and is the side that listens on
the comm port.

---

## 3. Launch

```bash
# node A
ssh A 'cd llama.cpp && GGML_RPC_PROFILE=256 \
  ./build-tp/bin/ggml-rpc-server -H 10.77.0.1 -p 50052 -c'

# node B
ssh B 'cd llama.cpp && GGML_RPC_PROFILE=256 \
  ./build-tp/bin/ggml-rpc-server -H 10.77.0.2 -p 50052 -c'
```

The comm port defaults to `port + 1000`, so `51052`. Override with `-C`.

Client, on node A:

```bash
./build-tp/bin/llama-bench --list-devices      # get the exact device names

./build-tp/bin/llama-bench -m /path/model.gguf \
  -rpc 10.77.0.1:50052,10.77.0.2:50052 \
  -sm tensor \
  --device 'RPC0[10.77.0.1:50052],RPC1[10.77.0.2:50052]' \
  -p 512 -n 128 -r 5
```

Pin `--device`. Otherwise the client's own Metal device is enumerated alongside the two
RPC ones and you get a three-way split, which `comm_init` rejects (`world != 2`).

`-sm tensor` is refused for some architectures; see `llm_arch_supports_sm_tensor` in
`src/llama-arch.cpp`.

---

## 4. Verify before trusting any measurement

Check these in order. Each one has silently produced a plausible but meaningless
benchmark at least once.

**RDMA is actually up.** On node B:

```
RDMA(Apple/UC) probed: dev=... ring=16 x 128 KiB
RDMA(Apple/UC) activated: qpn=...->... mtu=... rx_depth=...
```

You want **two** activations on B - one for the client link, one for the peer link.
Node A shows **one**: the client-to-A connection is A talking to its own Thunderbolt
address, which does not establish a QP and falls back to loopback TCP. That asymmetry is
expected.

**The frame size negotiated down on the peer link.** The comm link asks for 16 KiB, so
its probe line should read `ring=128 x 16 KiB`, while the client link stays
`ring=16 x 128 KiB`. If the peer link still says 128 KiB, the negotiation did not take
and every gate is padding a 28 KiB payload into 128 KiB.

**The gate channel came up.** A second queue pair carries the gate payloads:

```
RDMA(Apple/UC) gate channel up: qpn=...->...
```

**The channel armed, and whether the doorbell did.** This appears on the *second* gate of
a session, because the first goes over the byte stream and doubles as the arming barrier:

```
[gate_arm] gate channel armed for 28672 byte payloads, doorbell on
```

`doorbell off` means the fence words could not be registered - they are a Metal
allocation rather than the ggml one your `rdma_caps` run verified. Payloads still take
the fast path; only the release goes back through the host.
`gate channel unavailable` means one of the two ranks failed to arm, and both agreed to
stay on the byte stream. That is the designed outcome, not a fault, but it means none of
the gate-channel arms below are actually being measured.

**The communicator formed.**

```
[comm_init] device 0 joined pairwise comm as rank 0, fast sync on
```

`rank 1` on the other node. If you see `failed to connect to peer`, the addressing is
wrong - go back to section 2. If it says `fast sync off` while
`GGML_RPC_METAL_FAST_SYNC=1` is set, the fence kernel did not compile; look for
`coherent(system) is not supported here` and capture the compiler error above it.

---

## 5. Measurement arms

Run in this order. Each arm isolates one change, so a regression points somewhere.

| # | server env | what it isolates |
|---|---|---|
| 0 | `GGML_RPC_NO_RDMA=1`, `-sm layer` | plumbing only, no gates |
| 1 | `GGML_RPC_NO_RDMA=1` | gate cost with TCP as the bottleneck |
| 2 | *(none)* | **baseline** - RDMA, blocking gate |
| 3 | `GGML_RPC_METAL_FAST_SYNC=1` | the fence, one submission per gate |
| 4 | `GGML_RPC_METAL_FAST_SYNC=1 GGML_METAL_BATCH=1 GGML_RPC_NO_GATE_CHANNEL=1` | a token per submission, gates on the byte stream |
| 5 | `GGML_RPC_METAL_FAST_SYNC=1 GGML_METAL_BATCH=1 GGML_RPC_NO_DOORBELL=1` | exact-size gate channel, host still releases |
| 6 | `GGML_RPC_METAL_FAST_SYNC=1 GGML_METAL_BATCH=1` | **everything** - the peer's NIC releases the fence |

Environment goes on **both servers**, not the client. Restart both between arms.

The profiler prints every `GGML_RPC_PROFILE` gates:

```
rpc: allreduce decode/f32     256 gates: wait ... pack ... exch ... unpack ... submit ... us, ... KiB
rpc: allreduce prefill/bf16    18 gates: wait ... pack ... exch ... unpack ... submit ... us, ... KiB
```

Read the **`decode/f32`** row. That is the ~2-per-layer gate that runs on every token;
`prefill/bf16` is the large-payload path and behaves nothing like it.

What each column should do:

- `wait` and `submit` collapse in arm 3 - that is the fence.
- `pack` and `unpack` are near zero whenever the payload is host-visible.
- `exch` is the wire and should be roughly flat across arms 2-4. **If `exch` moves, be
  suspicious of the measurement**, not pleased.
- In arm 4 and later `pack`/`unpack`/`submit` are structurally zero: the payload moves
  straight to and from device memory and the reduce was encoded long before.
- Arm 5 should shave the residual framing: a 28 KiB payload moves 28 KiB instead of being
  rounded up to two 16 KiB frames.
- **Arm 6 is the one to watch.** `exch` stops being a round trip and becomes send-only,
  because the host no longer waits for the incoming partial - the peer's NIC lands it and
  then releases the GPU. If `exch` in arm 6 is not markedly below arm 5, the doorbell is
  not doing anything and the first thing to check is that the arming line said
  `doorbell on` on **both** nodes.

Also record, from the profile line, the **gates per token** (`gates` divided by tokens
generated). Everything scales with it and it has never been measured.

### Machine noise

Minima over many interleaved runs, not medians. A loaded machine has produced numbers
10x off on this hardware, in both directions. Close everything, check `uptime` on both
nodes first, and interleave arms rather than running each to completion.

---

## 6. Correctness gates

Performance numbers are worthless until these pass.

**Byte-identical greedy decode across all arms.** This is the check that catches a fence
releasing early:

```bash
for arm in "" "GGML_RPC_METAL_FAST_SYNC=1" "GGML_RPC_METAL_FAST_SYNC=1 GGML_METAL_BATCH=1"; do
  # restart both servers with $arm, then:
  ./build-tp/bin/llama-cli -m /path/model.gguf \
    -rpc 10.77.0.1:50052,10.77.0.2:50052 -sm tensor \
    --device 'RPC0[...],RPC1[...]' \
    --temp 0 --seed 1 -n 256 -no-cnv \
    -p "Write a haiku about Thunderbolt." > "out.$(echo $arm | md5).txt"
done
diff out.*.txt    # must be empty
```

Use a prompt long enough to cross the prefill/decode boundary, so the `wire_bf16` path
and the scratch-growth path both get exercised.

**Bounded failure, not a hang.** With arm 4 running, `kill -9` the server on B mid
generation. Expected on A:

- a `gate ... never arrived` or `exchange failed` error within 30 s,
- the connection dropped and the client aborting,
- **no** multi-minute GPU spin and no watchdog reset.

If the GPU sits at 100% for minutes, `GGML_RPC_FENCE_MAX_ITERS` is too high for this
hardware. It defaults to 8,000,000, which is about 4.4 s at the ~1.8 M iterations/sec
measured on an M1 Max - re-derive it on the parts you are actually running.

**Ungraceful client exit.** `kill -9` the client mid generation. Both servers should log
`Client connection closed` and accept a new connection. A crash here means the service
thread teardown is wrong.

**Scratch growth.** Run a short prompt then a much longer one against the same server
without restarting. This exercises the path that grows the comm scratch mid-stream, which
has to synchronise the GPU before unmapping the old pages. Note the gate channel arms
once for one payload size and registers the scratch at that moment; growing the scratch
replaces the memory the channel registered, so watch for a fallback or a failure here
specifically - this is the interaction I would expect to break first.

**Mixed payload sizes.** A prompt long enough to run prefill gates (bf16, large) and then
decode gates (f32, small) in the same session. Only one size gets the gate channel; the
other must fall back to the byte stream cleanly and produce identical output.

**Slot rotation under a peer that runs ahead.** Hard to force directly, but a long
generation with `GGML_RPC_PROFILE=64` and correctness checked at the end covers it: the
peer may run up to three gates ahead, and if the rotation is wrong the reduce reads a
partial that has already been overwritten. Wrong output with no error is the signature.

---

## 7. Environment reference

| variable | side | default | meaning |
|---|---|---|---|
| `GGML_RPC_METAL_FAST_SYNC` | server | off | GPU-side fence instead of a blocking wait per gate |
| `GGML_METAL_BATCH` | server | off | Metal defers submission; a token costs one submission |
| `GGML_RPC_NO_GATE_CHANNEL` | server | off | keep gates on the byte stream |
| `GGML_RPC_NO_DOORBELL` | server | off | keep the gate channel, host writes the release |
| `GGML_RPC_FENCE_MAX_ITERS` | server | 8000000 | bound on the GPU fence spin |
| `GGML_RPC_PROFILE` | server | off | report gate cost every N gates |
| `GGML_RPC_NO_RDMA` | either | off | force TCP |
| `GGML_RPC_NO_COMM` | client | off | route allreduce through the client instead of peer to peer |
| `GGML_RPC_DEBUG` | either | off | verbose RPC logging |

`GGML_METAL_BATCH` only helps a caller that submits many small dependent graphs. It gives
up the immediate submit of the first nodes and the multi-threaded encode of the rest, so
do not set it for ordinary single-node Metal work.

---

## 8. What is unproven

Validated on a single M1 Max, no network: the fence kernels, their ordering, the bounded
timeout, and 122 inline fences in one command buffer driven by a service thread.

**Never executed anywhere:**

- the RPC service thread against a real peer,
- the frame-size negotiation,
- the duplex gate exchange,
- the gate channel: second queue pair, registration, exact-size pre-posted receives, slot
  rotation, and the arming barrier,
- the doorbell,
- every failure path in section 6.

The gate channel and the doorbell are the largest untested surface, and their failure
modes are the quiet kind. A frame-count mismatch, a missed repost or a slot collision
shows up as a hang or as wrong output, not as an error - which is why the byte-identical
check in section 6 matters more than any timing number here.

Two structural costs are known and unfixed. Each per-gate `GRAPH_RECOMPUTE` and
`COMM_ALLREDUCE` from the client is a separate RPC message on the client link, which
still frames at 128 KiB - roughly 245 frames per token to carry about 36 KiB. And with
`GGML_METAL_BATCH=1` the whole token is encoded with the GPU idle, giving up overlap the
unbatched path gets for free. Counting `post_send` calls and bytes per token on both
sockets would turn the first of those from an estimate into a measurement, and is
probably the single most informative thing to add.
