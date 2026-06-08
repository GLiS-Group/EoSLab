#pragma once
/**
 * @file attributes.hpp
 * @brief Portable compiler-attribute macros used across the library.
 */

/**
 * @def GLIS_EOS_ALWAYS_INLINE
 * @brief Forces the compiler to inline the annotated function.
 *
 * Applied to the per-species kernels, the pre-calculations, and the CRTP
 * plumbing (`for_each_component`, `get_parameters`, the bulk Helmholtz
 * functions) that together make up the Helmholtz expression
 * [Enzyme](https://enzyme.mit.edu) differentiates. Forcing these to inline
 * collapses the whole expression into the differentiated entry point *before*
 * the Enzyme plugin runs, so the hoisted pre-calculation struct and the
 * per-species parameter struct never cross a call boundary as opaque aggregates.
 *
 * Without this, Enzyme's type/activity analysis cannot follow those aggregates
 * at low optimization levels (notably the `-O1` debug build) and silently
 * produces wrong derivatives, while `-O3` happens to inline + scalarize them
 * away. The attribute makes autodiff correctness independent of the optimization
 * level. Any new multi-fluid / one-fluid model should carry it on its kernels
 * and pre-calculations.
 *
 * Expands to `[[clang::always_inline]]` on Clang (the project's required
 * compiler, see the top-level CMakeLists) and to nothing otherwise.
 */
#if defined(__clang__)
#define GLIS_EOS_ALWAYS_INLINE [[clang::always_inline]]
#else
#define GLIS_EOS_ALWAYS_INLINE
#endif
