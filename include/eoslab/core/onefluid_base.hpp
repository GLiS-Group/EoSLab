#pragma once
/**
 * @file onefluid_base.hpp
 * @brief CRTP base that builds a full "one-fluid" EoS model from a mixing rule
 *        and a single bulk kernel.
 *
 * A one-fluid model collapses the mixture to a pseudo-pure fluid: a mixing rule
 * averages the per-species parameters (as functions of composition), then one
 * bulk expression is evaluated with those averaged parameters. Unlike a
 * multi-fluid model there is no per-species sum. This base owns the wiring
 * (call the mixing rule, pass its result to the bulk kernel) so an implementer
 * writes only the mixing rule and the bulk math.
 */

#include "eoslab/core/parameter_storage.hpp"

#include <concepts>
#include <cstddef>
#include <span>

namespace glis::eos {

/**
 * @brief CRTP base providing the bulk Helmholtz workflow for a one-fluid model.
 *
 * Inherits SoA parameter storage (construct it from a per-species list of
 * @p ParamStruct via the inherited constructors) and provides
 * `calc_helmholtz`, `calc_helmholtz_density`, and `calc_partial_helmholtz`,
 * which together satisfy glis::eos::EquationOfState.
 *
 * A derived model @p Derived must provide, all **public** (so the CRTP downcast
 * can call them):
 * - a **mixing rule** returning a `Number`-templated averaged-parameter struct
 *   (use `column(p)` for the contiguous, SIMD-friendly reduction over species):
 *   `perform_pre_calculations(c, x, T)` and `perform_pre_calculations(rho_i, T)`;
 * - the **bulk kernels**, taking the averaged struct (no species index):
 *   `calc_helmholtz_bulk(c, x, T, const Avg<Number>&)` and
 *   `calc_helmholtz_density_bulk(rho_i, T, const Avg<Number>&)`.
 *
 * @tparam Derived     The concrete model (CRTP).
 * @tparam ParamStruct Per-species parameter aggregate (all-`double`).
 * @tparam N           Component count, or @c std::dynamic_extent.
 *
 * @note `calc_helmholtz` / `calc_helmholtz_density` are the functions the core
 *       autodiff layer differentiates; the mixing rule depends on the active
 *       composition and so is evaluated inside them (part of the differentiated
 *       expression).
 */
template<class Derived, class ParamStruct, std::size_t N = std::dynamic_extent>
class OneFluidBase : public ParameterStorage<ParamStruct, N> {
public:
    using parameter_type = ParamStruct; ///< The per-species parameter struct.

    using ParameterStorage<ParamStruct, N>::ParameterStorage;

    /**
     * @brief Total molar Helmholtz energy of the pseudo-pure fluid.
     * @param c Molar concentration [mol/m^3].
     * @param x Mole-fraction array [-].
     * @param T Temperature [K].
     * @return Molar Helmholtz energy [J/mol].
     */
    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz(Number c, const Number* x, Number T) const
    {
        const auto avg = self().perform_pre_calculations(c, x, T);
        return self().calc_helmholtz_bulk(c, x, T, avg);
    }

    /**
     * @brief Total Helmholtz energy density of the pseudo-pure fluid.
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @return Helmholtz energy density [J/m^3].
     */
    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density(const Number* rho_i, Number T) const
    {
        const auto avg = self().perform_pre_calculations(rho_i, T);
        return self().calc_helmholtz_density_bulk(rho_i, T, avg);
    }

    /**
     * @brief Per-component Helmholtz energy density via mole-fraction split.
     *
     * A one-fluid model has no natural per-species decomposition, so the total
     * density is distributed by mole fraction: `out[i] = (rho_i / c) * Psi`. The
     * entries sum to @f$\Psi@f$. Provided for concept-completeness; chemical
     * potentials are obtained by autodiff of calc_helmholtz_density, not from
     * this split.
     *
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @param[out] out Per-component Helmholtz energy density [J/m^3]; length `size()`.
     */
    template<std::floating_point Number>
    void calc_partial_helmholtz(const Number* rho_i, Number T, Number* out) const
    {
        // A one-fluid model has no natural per-species split; distribute the
        // total density by mole fraction so the entries sum to Psi. (Chemical
        // potentials come from autodiff of calc_helmholtz_density, not this.)
        const Number psi = calc_helmholtz_density(rho_i, T);
        Number c{0};
        this->for_each_component([&](std::size_t i) { c += rho_i[i]; });
        this->for_each_component([&](std::size_t i) { out[i] = (rho_i[i] / c) * psi; });
    }

protected:
    [[nodiscard]] const Derived& self() const { return static_cast<const Derived&>(*this); }
};

} // namespace glis::eos
