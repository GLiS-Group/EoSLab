#pragma once

/**
 * @file eos_pair.hpp
 * @brief Couples an ideal and a residual model into a single equation of state.
 */

#include "eoslab/core/concepts.hpp"
namespace glis::eos {

/**
 * @brief A complete equation of state, formed by pairing an ideal contribution
 *        with a residual (departure) contribution.
 *
 * Thermodynamic properties are obtained by passing an instance of this class to
 * the free functions in core_calculations.hpp. The total reduced Helmholtz
 * energy is the sum of the ideal and residual parts, and the property routines
 * combine the two contributions as the relevant thermodynamics dictates.
 *
 * @tparam Ideal    A model satisfying glis::eos::IdealEoS.
 * @tparam Residual A model satisfying glis::eos::ResidualEoS.
 */
template<IdealEoS Ideal, ResidualEoS Residual> class EoS {
public:
    using ideal_type = Ideal;       ///< Type of the ideal contribution.
    using residual_type = Residual; ///< Type of the residual contribution.

    /**
     * @brief Construct from an ideal and a residual model.
     * @param ideal    The ideal contribution.
     * @param residual The residual contribution.
     * @pre  `ideal.size() == residual.size()` (checked via @c assert): both
     *       models must describe the same number of components.
     */
    EoS(Ideal ideal, Residual residual) noexcept(std::is_nothrow_move_constructible_v<Ideal> &&
                                                 std::is_nothrow_move_constructible_v<Residual>) :
        ideal_{std::move(ideal)}, residual_{std::move(residual)}
    {
        assert(ideal_.size() == residual_.size());
    }

    /// @brief Access the ideal contribution.
    const Ideal& ideal() const noexcept { return ideal_; }
    /// @brief Access the residual contribution.
    const Residual& residual() const noexcept { return residual_; }
    /**
     * @brief Number of chemical components.
     * @return Component count (equal for both contributions).
     */
    [[nodiscard]] constexpr std::size_t size() const noexcept { return ideal_.size(); }

private:
    Ideal ideal_;       ///< The ideal contribution.
    Residual residual_; ///< The residual contribution.
};
} // namespace glis::eos