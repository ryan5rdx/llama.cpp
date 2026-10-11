#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif

#include <metal_stdlib>
using namespace metal;
namespace metal { constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }

// tensor-parallel all-reduce gate. threadgroup 0 publishes this rank's arrival
// (words[0] = seq) once the graph's stores are system-visible, every threadgroup
// spins on the host's release (words[1] >= seq), then all fold dst += peer.
// words[2] latches a timeout (the reduce is skipped); words[3] reports the spin
// iteration count of threadgroup 0.
kernel void ggml_metal_gate_wait_reduce_f32(
        device         float * dst  [[buffer(0)]],
        device   const float * peer [[buffer(1)]],
        volatile coherent(system) device uint * words [[buffer(2)]],
        constant uint & seq   [[buffer(3)]],
        constant uint & maxit [[buffer(4)]],
        constant uint & n     [[buffer(5)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threadgroups_per_grid]],
        uint tptg  [[threads_per_threadgroup]]) {
    threadgroup uint released;
    if (tpitg == 0) {
        if (tgpig == 0) {
            metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
            words[0] = seq;
            metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
        }
        uint i = 0;
        for (; i < maxit; ++i) {
            metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
            if (words[1] >= seq) break;
        }
        released = (i < maxit) ? 1u : 0u;
        if (tgpig == 0) {
            words[3] = i;
            if (released == 0u) { words[2] = 1u; }
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (released == 0u) { return; }
    for (uint i = tgpig*tptg + tpitg; i < n; i += ntg*tptg) {
        dst[i] += peer[i];
    }
}
