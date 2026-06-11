#pragma once
/**
 * @file attributes.hpp
 * @brief Portable compiler-attribute macros used across the library.
 */

/**
 * @def GLIS_EOS_ALWAYS_INLINE
 * @brief Forces the compiler to inline the annotated function.
 *
 * Applied to the small helpers (e.g., `eval_polynomial`) that
 * sit inside the Helmholtz expression [Enzyme](https://enzyme.mit.edu)
 * differentiates. Forcing these to inline collapses the whole expression into
 * the differentiated entry point *before* the Enzyme plugin runs, so nothing
 * crosses a call boundary as an opaque aggregate.
 *
 * Without this, Enzyme's type/activity analysis cannot follow such aggregates
 * at low optimization levels (notably the `-O1` debug build) and silently
 * produces wrong derivatives, while `-O3` happens to inline + scalarize them
 * away. The attribute makes autodiff correctness independent of the
 * optimization level; any new helper called from a model's Helmholtz kernels
 * should carry it.
 *
 * Expands to `[[clang::always_inline]]` on Clang (the project's required
 * compiler, see the top-level CMakeLists) and to nothing otherwise.
 */
#if defined(__clang__)
#define GLIS_EOS_ALWAYS_INLINE [[clang::always_inline]]
#else
#define GLIS_EOS_ALWAYS_INLINE
#endif
