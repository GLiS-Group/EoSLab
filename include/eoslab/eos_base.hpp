#pragma once

/**
 * @file eos_base.hpp
 * @brief Base classes that carry the component-count of an equation of state and
 *        tag a model as an ideal contribution.
 */

#include <cassert>
#include <cstddef>
#include <span>
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