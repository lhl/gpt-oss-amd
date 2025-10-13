#pragma once

// Lightweight no-op profiling hooks to satisfy references in the build.
// These can be replaced with real implementations later.
static inline void set_profiling_enabled(bool /*enabled*/) {}
static inline void reset_batch_timings() {}
static inline void print_batch_timing_summary() {}

