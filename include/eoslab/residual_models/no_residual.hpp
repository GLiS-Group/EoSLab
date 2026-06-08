#pragma once

#include "eoslab/core/concepts.hpp"

#include <span>

namespace glis::eos {

/**
 * @brief A residual contribution that is identically zero.
 *
 * Every thermodynamic quantity this model contributes evaluates to zero, so
 * pairing it (via glis::eos::EoS) with an ideal-gas model yields a complete
 * equation of state whose residual (departure) part vanishes &mdash; i.e. a pure
 * ideal gas. It exists so that ideal-gas behaviour can be expressed through the
 * same EoS machinery as any other model without a special-case code path.
 *
 * @tparam N Number of components, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N> class NoResidual : public BaseEoS<N> {
public:
    /**
     * @brief Default-construct a compile-time-sized model.
     *
     * Only available when the component count @p N is known at compile time
     * (i.e. @p N is not @c std::dynamic_extent); the size is carried by the type.
     */
    constexpr NoResidual() noexcept
        requires(N != std::dynamic_extent)
    = default;

    /**
     * @brief Construct a runtime-sized model with an explicit component count.
     *
     * Only available when @p N is @c std::dynamic_extent; the component count is
     * not known until construction and is therefore supplied here.
     *
     * @param n Number of chemical components.
     */
    constexpr explicit NoResidual(const std::size_t n) noexcept
        requires(N == std::dynamic_extent)
        : BaseEoS<N>(n)
    {
    }

    /**
     * @brief Molar residual Helmholtz energy, which is identically zero.
     * @return @c Number{0}.
     */
    template<std::floating_point Number> Number calc_helmholtz(const Number /*c*/, const Number* /*x*/, const Number /*T*/) const
    {
        return Number{0};
    }

    /**
     * @brief Total residual Helmholtz energy density, which is identically zero.
     * @return @c Number{0}.
     */
    template<std::floating_point Number>
    Number calc_helmholtz_density(const Number* /*rho_i*/, const Number /*T*/) const
    {
        return Number{0};
    }

    /**
     * @brief Per-component residual Helmholtz energy density, all zero.
     *
     * Writes @c Number{0} into each of the @c size() entries of @p out.
     *
     * @param out Output array; filled with zeros.
     */
    template<std::floating_point Number>
    void calc_partial_helmholtz(const Number* /*rho_i*/, const Number /*T*/, Number* out) const
    {
        this->for_each_component([&](std::size_t idx) { out[idx] = Number{0}; });
    }
};

static_assert(ResidualEoS<NoResidual<2>>, "NoResidual must satisfy the ResidualEoS concept.");
static_assert(ResidualEoS<NoResidual<std::dynamic_extent>>, "NoResidual must satisfy the ResidualEoS concept.");

} // namespace glis::eos