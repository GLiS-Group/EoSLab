#pragma once

/**
 * @file eos_base.hpp
 * @brief Base classes that carry the component-count of an equation of state and
 *        tag a model as an ideal contribution.
 */

#include "eoslab/core/attributes.hpp"

#include <cassert>
#include <cstddef>
#include <span>
#include <utility>
namespace glis::eos {

/**
 * @brief Base class that stores the number of chemical components an EoS
 *        describes, either at compile time or at run time.
 *
 * Provide a concrete component count as the template argument to bake the size
 * into the type (zero storage, `size()` is `static constexpr`). Use the default
 * @c std::dynamic_extent for a size only known at run time (see the
 * specialization below).
 *
 * @tparam N Number of components, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent> class BaseEoS {
public:
    constexpr BaseEoS() noexcept = default;

    /**
     * @brief Construct with an explicit component count.
     * @param n Number of components. Must equal @p N (checked via @c assert).
     */
    constexpr explicit BaseEoS([[maybe_unused]] const std::size_t n) { assert(n == N); }

    /**
     * @brief Number of chemical components.
     * @return @p N
     */
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

    /**
     * @brief Invoke @p f once for every component index, in ascending order.
     *
     * Calls `f(0), f(1), ..., f(size() - 1)`. For a compile-time component count
     * the loop is expanded as a fold over @c std::index_sequence, so the bound is
     * known at compile time and the body is fully unrolled. This lets a model
     * write a single component loop instead of branching on whether the size is
     * static or dynamic.
     *
     * @tparam F Callable invocable as `f(std::size_t)`.
     * @param  f Callable applied to each component index. Taken by value (like
     *           the standard algorithms), so a mutable callable may carry state
     *           across the visits.
     *
     * @code{.cpp}
     * // Zero every component of an output buffer:
     * model.for_each_component([&](std::size_t i) { out[i] = 0.0; });
     * @endcode
     */
    template<class F> GLIS_EOS_ALWAYS_INLINE constexpr void for_each_component(F f) const
    {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) { (f(Is), ...); }(std::make_index_sequence<N>{});
    }
};

/**
 * @brief Runtime-sized specialization of BaseEoS.
 *
 * Stores the component count in a data member because it is not known until
 * construction.
 */
template<> class BaseEoS<std::dynamic_extent> {
public:
    /**
     * @brief Construct with the runtime component count.
     * @param n Number of components.
     */
    constexpr explicit BaseEoS(const std::size_t n) noexcept : n_(n) {}

    /**
     * @brief Number of chemical components.
     * @return The stored component count.
     */
    [[nodiscard]] constexpr std::size_t size() const noexcept { return n_; }

    /**
     * @brief Invoke @p f once for every component index, in ascending order.
     *
     * Calls `f(0), f(1), ..., f(size() - 1)` via a runtime loop over the stored
     * component count. Mirrors the compile-time-sized overload so models can use
     * the same component-loop spelling regardless of whether the size is known
     * at compile time.
     *
     * @tparam F Callable invocable as `f(std::size_t)`.
     * @param  f Callable applied to each component index. Taken by value (like
     *           the standard algorithms), so a mutable callable may carry state
     *           across the visits.
     */
    template<class F> GLIS_EOS_ALWAYS_INLINE constexpr void for_each_component(F f) const
    {
        for (std::size_t idx = 0; idx < n_; ++idx) {
            f(idx);
        }
    }

private:
    std::size_t n_; ///< Number of components.
};

/**
 * @brief Empty tag type that marks a model as an *ideal* equation of state.
 *
 * A model must publicly derive from this class to satisfy the glis::eos::IdealEoS
 * concept. It carries no data or behaviour; it exists purely so the concepts can
 * distinguish ideal contributions from residual ones at compile time.
 */
class BaseIdealEoS {};
} // namespace glis::eos