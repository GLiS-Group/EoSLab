#pragma once
/**
 * @file multifluid_base.hpp
 * @brief CRTP base that builds a full "multi-fluid" EoS model from per-species
 *        kernels.
 *
 * A multi-fluid model is one whose total Helmholtz energy is a sum of
 * per-species contributions. This base owns the shared workflow &mdash; the
 * component loop, the optional pre-calculation (computed once and hoisted out of
 * the loop), and the parameter lookup &mdash; so an implementer writes only the
 * per-species math and a parameter struct.
 */

#include "eoslab/core/attributes.hpp"
#include "eoslab/core/parameter_storage.hpp"

#include <concepts>
#include <cstddef>
#include <span>

namespace glis::eos {

namespace detail {
/// @internal Does @p D define a pre-calculation for the (c, x, T) signature?
template<class D, class Number>
concept has_molar_pre_calc =
    requires(const D& d, Number c, const Number* x, Number T) { d.perform_pre_calculations(c, x, T); };
/// @internal Does @p D define a pre-calculation for the (rho_i, T) signature?
template<class D, class Number>
concept has_density_pre_calc =
    requires(const D& d, const Number* rho_i, Number T) { d.perform_pre_calculations(rho_i, T); };
} // namespace detail

/**
 * @brief CRTP base providing the bulk Helmholtz workflow for a multi-fluid model.
 *
 * Inherits SoA parameter storage (so construct it from a per-species list of
 * @p ParamStruct via the inherited constructors) and provides
 * `calc_helmholtz`, `calc_helmholtz_density`, and `calc_partial_helmholtz`,
 * which together satisfy glis::eos::EquationOfState.
 *
 * A derived model @p Derived must provide two **public** per-species kernels
 * (so the CRTP downcast can call them):
 * - `calc_helmholtz_i(c, x, T, i, const ParamStruct&)` &rarr; species @p i 's
 *   contribution to the molar Helmholtz energy.
 * - `calc_helmholtz_density_i(rho_i, T, i, const ParamStruct&)` &rarr; species
 *   @p i 's contribution to the Helmholtz energy density.
 *
 * Optionally, a model may define a **pre-calculation** &mdash; a value computed
 * once and reused for every species (e.g. caching @f$\ln T@f$ or a mole-fraction
 * sum). To opt in, define both overloads, returning a `Number`-templated struct:
 * - `perform_pre_calculations(c, x, T)` and `perform_pre_calculations(rho_i, T)`,
 * and add a trailing `const Pre<Number>&` parameter to the two kernels. The base
 * detects these automatically; a model without them simply omits the function
 * and the extra kernel argument.
 *
 * @tparam Derived     The concrete model (CRTP).
 * @tparam ParamStruct Per-species parameter aggregate (all-`double`); its arity
 *                     is the number of parameters per species.
 * @tparam N           Component count, or @c std::dynamic_extent.
 *
 * @note `calc_helmholtz` / `calc_helmholtz_density` are the functions the core
 *       autodiff layer differentiates; the pre-calculation (which depends on the
 *       active inputs) is therefore computed *inside* them so it is part of the
 *       differentiated expression.
 */
template<class Derived, class ParamStruct, std::size_t N = std::dynamic_extent>
class MultiFluidBase : public ParameterStorage<ParamStruct, N> {
public:
    using parameter_type = ParamStruct; ///< The per-species parameter struct.

    using ParameterStorage<ParamStruct, N>::ParameterStorage;

    /**
     * @brief Total molar Helmholtz energy @f$a = \sum_i a_i@f$.
     * @param c Molar concentration [mol/m^3].
     * @param x Mole-fraction array [-].
     * @param T Temperature [K].
     * @return Molar Helmholtz energy [J/mol].
     */
    template<std::floating_point Number> [[nodiscard]] Number calc_helmholtz(Number c, const Number* x, Number T) const
    {
        Number a{0};
        if constexpr (detail::has_molar_pre_calc<Derived, Number>) {
            const auto pre = self().perform_pre_calculations(c, x, T);
            this->for_each_component(
                [&](std::size_t i) { a += self().calc_helmholtz_i(c, x, T, i, this->get_parameters(i), pre); });
        }
        else {
            this->for_each_component(
                [&](std::size_t i) { a += self().calc_helmholtz_i(c, x, T, i, this->get_parameters(i)); });
        }
        return a;
    }

    /**
     * @brief Total Helmholtz energy density @f$\Psi = \sum_i \Psi_i@f$.
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @return Helmholtz energy density [J/m^3].
     */
    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density(const Number* rho_i, Number T) const
    {
        Number psi{0};
        if constexpr (detail::has_density_pre_calc<Derived, Number>) {
            const auto pre = self().perform_pre_calculations(rho_i, T);
            this->for_each_component([&](std::size_t i) {
                psi += self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i), pre);
            });
        }
        else {
            this->for_each_component(
                [&](std::size_t i) { psi += self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i)); });
        }
        return psi;
    }

    /**
     * @brief Per-component Helmholtz energy density, @f$\text{out}[i] = \Psi_i@f$.
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @param[out] out Per-component Helmholtz energy density [J/m^3]; length `size()`.
     */
    template<std::floating_point Number> void calc_partial_helmholtz(const Number* rho_i, Number T, Number* out) const
    {
        if constexpr (detail::has_density_pre_calc<Derived, Number>) {
            const auto pre = self().perform_pre_calculations(rho_i, T);
            this->for_each_component([&](std::size_t i) {
                out[i] = self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i), pre);
            });
        }
        else {
            this->for_each_component(
                [&](std::size_t i) { out[i] = self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i)); });
        }
    }

protected:
    [[nodiscard]] const Derived& self() const { return static_cast<const Derived&>(*this); }
};

} // namespace glis::eos
