#pragma once
/**
 * @file core_calculations.hpp
 * @brief Thermodynamic property calculations built on top of an EoS pair.
 *
 * Every property is derived from the reduced molar Helmholtz energy
 * @f$\alpha = a/(RT)@f$ and its derivatives. The derivatives are obtained by
 * automatic differentiation with [Enzyme](https://enzyme.mit.edu):
 * - forward mode for the @f$1/T@f$- and @f$c@f$-derivatives (see
 *   detail::calc_alpha / detail::calc_lambda), and
 * - reverse mode for the partial-molar derivatives w.r.t. each @f$\rho_i@f$
 *   (see detail::calc_dPsi_drhoi), used for chemical potentials and fugacities.
 *
 * Symbol / unit conventions used throughout:
 * - `c`     molar concentration (molar density) [mol/m^3]
 * - `x`     mole fractions [-]
 * - `T`     temperature [K]
 * - `invT`  inverse temperature @f$1/T@f$ [1/K]
 * - `rho_i` partial molar concentrations [mol/m^3]
 * - `R`     gas constant [J/(mol K)]
 */
#include "eoslab/core/concepts.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/core/numbers.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

// NOLINTBEGIN
/// @cond INTERNAL
// Enzyme autodiff requires a few global definitions
// TODO: Make these `const` to please clangd?? Try making them const after writing tests to see.
inline int enzyme_dup;
inline int enzyme_dupnoneed;
inline int enzyme_out;
inline int enzyme_const;

template<typename return_type, typename... T> return_type __enzyme_fwddiff(void*, T...);

template<typename return_type, typename... T> return_type __enzyme_autodiff(void*, T...);
/// @endcond
// NOLINTEND

namespace glis::eos {

namespace detail {
/**
 * @internal
 * @brief Compile-time integer power @f$\text{base}^N@f$ by exponentiation by
 *        squaring.
 *
 * @tparam Number Floating-point base type.
 * @tparam N      Integer exponent. May be negative (computes the reciprocal
 *                power) or zero (returns 1).
 * @param  base   The base. Units are arbitrary; the result carries `base`'s unit
 *                raised to the @p N.
 * @return @f$\text{base}^N@f$.
 * @pre    For @p N < 0, `base != 0`.
 */
template<std::floating_point Number, int N> constexpr Number fast_pow(const Number base)
{
    if constexpr (N == 0) {
        return Number{1};
    }
    else if constexpr (N < 0) {
        assert(base != Number{0});
        return fast_pow<Number, -N>(Number{1} / base);
    }
    else if constexpr (N == 1) {
        return base;
    }
    else if constexpr (N % 2 == 0) {
        Number half = fast_pow<Number, N / 2>(base);
        return half * half;
    }
    else {
        return base * fast_pow<Number, N - 1>(base);
    }
}

/**
 * @internal
 * @brief Mixed derivative of the reduced molar Helmholtz energy
 *        @f$\alpha = a/(RT)@f$.
 *
 * Returns
 * @f$\dfrac{\partial^{\,i+j}\alpha}{\partial (1/T)^{i}\,\partial c^{j}}@f$,
 * evaluated by recursively applying Enzyme forward-mode differentiation: an
 * @f$1/T@f$-derivative while `i > j`, otherwise a `c`-derivative. The base case
 * (`i == j == 0`) is @f$\alpha = a/(RT)@f$ itself.
 *
 * @tparam i    Number of derivatives w.r.t. inverse temperature @f$1/T@f$.
 * @tparam j    Number of derivatives w.r.t. concentration `c`.
 * @param  eos  A single-contribution model (ideal or residual).
 * @param  c    Molar concentration [mol/m^3].
 * @param  x    Mole-fraction array [-].
 * @param  invT Inverse temperature @f$1/T@f$ [1/K].
 * @return The derivative; @f$\alpha@f$ is dimensionless, so the result has units
 *         @f$\mathrm{K}^{i}\,(\mathrm{m^3/mol})^{j}@f$.
 */
template<int i, int j, EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_alpha(const EoS& eos, const Number c, const Number* x, const Number invT)
{
    static_assert(i >= 0, "The template parameter `i` must be non-negative. It represent the number of derivatives "
                          "with respect to `invT`.");
    static_assert(
        j >= 0,
        "The template parameter `j` must be non-negative. It represent the number of derivatives with respect to `c`.");
    if constexpr (i == 0 && j == 0) {
        const Number T = Number{1} / invT;
        constexpr Number R = ideal_gas_constant<Number>;
        return eos.calc_helmholtz(c, x, T) / (R * T);
    }
    else if constexpr (i > j) {
        Number dinvT{1.};
        return __enzyme_fwddiff<Number>((void*)calc_alpha<i - 1, j, EoS, Number>, enzyme_const, &eos, enzyme_const, c,
                                        enzyme_const, x, enzyme_dup, invT, dinvT);
    }
    else {
        Number dc{1.};
        return __enzyme_fwddiff<Number>((void*)calc_alpha<i, j - 1, EoS, Number>, enzyme_const, &eos, enzyme_dup, c, dc,
                                        enzyme_const, x, enzyme_const, invT);
    }
}

/**
 * @internal
 * @brief Dimensionless scaled derivative
 *        @f$\lambda_{i,j} = (1/T)^{i}\,c^{\,j}\,
 *        \dfrac{\partial^{\,i+j}\alpha}{\partial (1/T)^{i}\,\partial c^{j}}@f$.
 *
 * Multiplying detail::calc_alpha by @f$(1/T)^i c^j@f$ cancels the units of the
 * derivative, giving the dimensionless reduced derivatives that the property
 * formulas below are written in terms of.
 *
 * @tparam i    Number of @f$1/T@f$-derivatives.
 * @tparam j    Number of `c`-derivatives.
 * @param  eos  A single-contribution model (ideal or residual).
 * @param  c    Molar concentration [mol/m^3].
 * @param  x    Mole-fraction array [-].
 * @param  invT Inverse temperature @f$1/T@f$ [1/K].
 * @return @f$\lambda_{i,j}@f$ [-] (dimensionless).
 */
template<int i, int j, EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_lambda(const EoS& eos, const Number c, const Number* x, const Number invT)
{
    // TODO: better assertions?
    static_assert(i >= 0, "The template parameter `i` must be non-negative. It represent the number of derivatives "
                          "with respect to `invT`.");
    static_assert(
        j >= 0,
        "The template parameter `j` must be non-negative. It represent the number of derivatives with respect to `c`.");
    return fast_pow<Number, i>(invT) * fast_pow<Number, j>(c) * calc_alpha<i, j, EoS, Number>(eos, c, x, invT);
}

/**
 * @internal
 * @brief Total Helmholtz energy density @f$\Psi = \sum_i \Psi_i@f$ for one
 *        contribution.
 *
 * Thin wrapper over the model's `calc_helmholtz_density`, which accumulates the
 * sum directly into a scalar (no per-component buffer). This is the scalar that
 * detail::calc_dPsi_drhoi differentiates (reverse mode) to obtain chemical
 * potentials.
 *
 * @param  eos   A single-contribution model (ideal or residual).
 * @param  rho_i Partial molar concentrations [mol/m^3].
 * @param  T     Temperature [K].
 * @return Total Helmholtz energy density @f$\Psi@f$ [J/m^3].
 */
template<EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_Psi(const EoS& eos, const Number* GLIS_EOS_RESTRICT rho_i, const Number T)
{
    return eos.calc_helmholtz_density(rho_i, T);
}

/**
 * @internal
 * @brief @p i-th temperature derivative of the Helmholtz energy density,
 *        @f$\partial^{i}\Psi/\partial T^{i}@f$.
 *
 * Computed by recursively applying Enzyme forward-mode differentiation w.r.t.
 * @p T; the base case (`i == 0`) is detail::calc_Psi.
 *
 * @tparam i      Number of temperature derivatives.
 * @param  eos    A single-contribution model (ideal or residual).
 * @param  rho_i  Partial molar concentrations [mol/m^3].
 * @param  T      Temperature [K].
 * @return @f$\partial^{i}\Psi/\partial T^{i}@f$ [J/(m^3 K^i)].
 */
template<int i, EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_dPsi_dT(const EoS& eos, const Number* GLIS_EOS_RESTRICT rho_i, const Number T)
{
    static_assert(i >= 0, "The template parameter `i` must be non-negative. It represent the number of derivatives "
                          "with respect to `T`.");
    if constexpr (i == 0) {
        return calc_Psi(eos, rho_i, T);
    }
    else {
        Number dT{1};
        return __enzyme_fwddiff<Number>((void*)calc_dPsi_dT<i - 1, EoS, Number>, enzyme_const, &eos, enzyme_const,
                                        rho_i, enzyme_dup, T, dT);
    }
}

/**
 * @internal
 * @brief Gradient of the Helmholtz energy density w.r.t. the partial molar
 *        concentrations, @f$\partial\Psi/\partial\rho_i@f$ (a chemical potential).
 *
 * Uses Enzyme reverse mode on detail::calc_Psi. Currently only the first
 * derivative (`i == 1`) is implemented; higher orders would require tensors.
 *
 * @warning Enzyme reverse mode **accumulates** into @p dPsi_drho. The caller must
 *          zero it beforehand; calling repeatedly with the same buffer sums the
 *          contributions (this is how the ideal and residual parts are combined).
 *
 * @tparam i        Derivative order; must be 1.
 * @param  eos      A single-contribution model (ideal or residual).
 * @param  rho_i    Partial molar concentrations [mol/m^3].
 * @param  T        Temperature [K].
 * @param  dPsi_drho Output gradient (length `eos.size()`), accumulated [J/mol].
 */
template<int i, EquationOfState EoS, std::floating_point Number>
void calc_dPsi_drhoi(const EoS& eos, const Number* GLIS_EOS_RESTRICT rho_i, const Number T,
                     Number* GLIS_EOS_RESTRICT dPsi_drho)
{
    // TODO: only implemented for 1 derivative because higher order would require tensors!
    static_assert(i > 0, "The template parameter `i` must be positive. It represents the number of derivatives with "
                         "respect to each `rho_i`. Use `calc_Psi()` for the '0th' derivative.");
    // NOTE: Enzyme reverse mode ACCUMULATES into `dPsi_drho`, so the caller must
    // zero it before the first call (and may sum several contributions by
    // calling repeatedly with the same buffer).
    if constexpr (i == 1) {
        // `calc_Psi` accumulates the per-component Helmholtz density into a local
        // scalar, so there is no caller-visible intermediate buffer to shadow:
        // `rho_i` is the only active input and the gradient lands in `dPsi_drho`.
        //
        // The model `eos` is logically constant, but Enzyme's activity analysis
        // fails to prove that the parameter storage read through it is inactive
        // once that storage is large enough (~16+ doubles): it then computes a
        // gradient w.r.t. the parameters and, with no shadow provided for an
        // `enzyme_const` argument, writes that gradient in place — clobbering the
        // model's parameters. We therefore pass `eos` as `enzyme_dup` with a
        // zero-initialized scratch shadow so the (unwanted) parameter gradient is
        // accumulated there and discarded, leaving the real parameters intact.
        // alignas(EoS) std::byte eos_shadow[sizeof(EoS)] = {};
        // __enzyme_autodiff<Number>((void*)calc_Psi<EoS, Number>, enzyme_dup, &eos,
        //                           reinterpret_cast<EoS*>(static_cast<void*>(eos_shadow)), enzyme_dup, rho_i, dPsi_drho,
        //                           enzyme_const, T);
        __enzyme_autodiff<Number>((void*)calc_Psi<EoS, Number>, enzyme_const, &eos, enzyme_dup, rho_i, dPsi_drho,
                                  enzyme_const, T);
        return;
    }
    else {
        static_assert(false, "Higher order derivatives are not implemented (yet?) because they require tensors!");
        return;
    }
}

/**
 * @internal
 * @brief Forward-mode alternative to detail::calc_dPsi_drhoi: the gradient
 *        @f$\partial\Psi/\partial\rho_i@f$ computed with one Enzyme *forward*-mode
 *        pass per component instead of a single reverse-mode pass.
 *
 * For a scalar output @f$\Psi(\rho)@f$ a forward-mode pass seeded with the unit
 * direction @f$e_j@f$ returns exactly @f$\partial\Psi/\partial\rho_j@f$, so the
 * full gradient is recovered with `eos.size()` passes. This is the natural
 * comparison point for the reverse-mode routine: reverse computes the whole
 * gradient in one pass, forward needs @f$n@f$ passes, and which is faster depends
 * on @f$n@f$ and on how well Enzyme handles each mode for a given model.
 *
 * Unlike the reverse-mode routine, the model @p eos is passed as
 * @c enzyme_const: in forward mode a constant input simply carries a zero tangent
 * and is never written, so no parameter-gradient scratch shadow is required.
 *
 * @warning Like the reverse-mode routine this **accumulates** into @p dPsi_drho,
 *          so the caller must zero it first (allowing the ideal and residual
 *          contributions to be summed in place).
 *
 * @param  eos       A single-contribution model (ideal or residual).
 * @param  rho_i     Partial molar concentrations [mol/m^3].
 * @param  T         Temperature [K].
 * @param  dPsi_drho Output gradient (length `eos.size()`), accumulated [J/mol].
 * @param  seed      Scratch direction buffer (length `eos.size()`); must be all
 *                   zero on entry and is left all zero on return.
 */
template<EquationOfState EoS, std::floating_point Number>
void calc_dPsi_drhoi_fwd(const EoS& eos, const Number* GLIS_EOS_RESTRICT rho_i, const Number T,
                         Number* GLIS_EOS_RESTRICT dPsi_drho, Number* GLIS_EOS_RESTRICT seed)
{
    const std::size_t n = eos.size();
    for (std::size_t j = 0; j < n; ++j) {
        seed[j] = Number{1};
        dPsi_drho[j] += __enzyme_fwddiff<Number>((void*)calc_Psi<EoS, Number>, enzyme_const, &eos, enzyme_dup, rho_i,
                                                 seed, enzyme_const, T);
        seed[j] = Number{0};
    }
}

/**
 * @internal
 * @brief Allocate a zeroed direction-seed buffer and accumulate the forward-mode
 *        gradient (calc_dPsi_drhoi_fwd) into @p out.
 *
 * Picks stack storage for a compile-time component count and a heap buffer for a
 * runtime count, then forwards to detail::calc_dPsi_drhoi_fwd. Factored out so
 * the public `*_fwd` property routines do not each repeat the seed bookkeeping.
 */
template<EquationOfState EoS, std::floating_point Number, std::size_t N>
void accumulate_dPsi_drhoi_fwd(const EoS& eos, std::span<const Number, N> rho_i, const Number T, Number* out)
{
    if constexpr (N == std::dynamic_extent) {
        std::vector<Number> seed(rho_i.size(), Number{0});
        calc_dPsi_drhoi_fwd(eos, rho_i.data(), T, out, seed.data());
    }
    else {
        std::array<Number, N> seed{};
        calc_dPsi_drhoi_fwd(eos, rho_i.data(), T, out, seed.data());
    }
}
} // namespace detail

/**
 * @brief Total molar Helmholtz energy @f$a = a^{\text{ideal}} + a^{\text{res}}@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Molar Helmholtz energy [J/mol].
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_helmholtz(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    return ideal.calc_helmholtz(c, x.data(), T) + residual.calc_helmholtz(c, x.data(), T);
}

/**
 * @brief Pressure @f$p = cRT\,(1 + \lambda^{\text{res}}_{0,1})@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Pressure [Pa].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_pressure(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return c * R * T * (Number{1} + detail::calc_lambda<0, 1>(residual, c, x.data(), invT));
}

/**
 * @brief Molar internal energy @f$u = a + Ts@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Molar internal energy [J/mol].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_internal_energy(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x,
                            const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * T *
           (detail::calc_lambda<1, 0>(ideal, c, x.data(), invT) +
            detail::calc_lambda<1, 0>(residual, c, x.data(), invT));
}

/**
 * @brief Molar enthalpy @f$h = u + p/c@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Molar enthalpy [J/mol].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_enthalpy(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * T *
           (Number{1} + detail::calc_lambda<0, 1>(residual, c, x.data(), invT) +
            detail::calc_lambda<1, 0>(ideal, c, x.data(), invT) +
            detail::calc_lambda<1, 0>(residual, c, x.data(), invT));
}

/**
 * @brief Molar entropy @f$s = - (\partial a / \partial T)_v@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Molar entropy [J/(mol K)].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_entropy(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * (detail::calc_lambda<1, 0>(ideal, c, x.data(), invT) +
                detail::calc_lambda<1, 0>(residual, c, x.data(), invT) -
                detail::calc_lambda<0, 0>(ideal, c, x.data(), invT) -
                detail::calc_lambda<0, 0>(residual, c, x.data(), invT));
}

/**
 * @brief Molar Gibbs energy @f$g = h - Ts@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Molar Gibbs energy [J/mol].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_gibbs(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * T *
           (Number{1} + detail::calc_lambda<0, 1>(residual, c, x.data(), invT) +
            detail::calc_lambda<0, 0>(ideal, c, x.data(), invT) +
            detail::calc_lambda<0, 0>(residual, c, x.data(), invT));
}

/**
 * @brief Partial derivative of pressure w.r.t. concentration,
 *        @f$(\partial p/\partial c)_{T,x}@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return @f$\partial p/\partial c@f$ [Pa m^3/mol] (= J/mol).
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_dp_dc(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * T *
           (Number{1} + (Number{2} * detail::calc_lambda<0, 1>(residual, c, x.data(), invT)) +
            detail::calc_lambda<0, 2>(residual, c, x.data(), invT));
}

/**
 * @brief Partial derivative of pressure w.r.t. temperature,
 *        @f$(\partial p/\partial T)_{c,x}@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return @f$\partial p/\partial T@f$ [Pa/K].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_dp_dT(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * c *
           (Number{1} + detail::calc_lambda<0, 1>(residual, c, x.data(), invT) -
            detail::calc_lambda<1, 1>(residual, c, x.data(), invT));
}

/**
 * @brief Molar isochoric heat capacity @f$c_v = (\partial u/\partial T)_{c}@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Molar @f$c_v@f$ [J/(mol K)].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_cv(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return -R * (detail::calc_lambda<2, 0>(ideal, c, x.data(), invT) +
                 detail::calc_lambda<2, 0>(residual, c, x.data(), invT));
}

/**
 * @brief Molar isobaric heat capacity @f$c_p@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @return Molar @f$c_p@f$ [J/(mol K)].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_cp(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    const Number lr_01 = detail::calc_lambda<0, 1>(residual, c, x.data(), invT);
    return R *
           (-detail::calc_lambda<2, 0>(ideal, c, x.data(), invT) -
            detail::calc_lambda<2, 0>(residual, c, x.data(), invT) +
            (detail::fast_pow<Number, 2>(Number{1} + lr_01 - detail::calc_lambda<1, 1>(residual, c, x.data(), invT)) /
             (Number{1} + (Number{2} * lr_01) + detail::calc_lambda<0, 2>(residual, c, x.data(), invT))));
}

/**
 * @brief Squared speed of sound @f$w^2 = c_p\,(\partial p/\partial c)/(M\,c_v)@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @param effective_molar_mass Mixture molar mass @f$M@f$ [kg/mol].
 * @return Squared speed of sound [m^2/s^2].
 * @pre   `T > 0`.
 * @pre `x.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_sound_speed_squared(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x,
                                const Number T, const Number effective_molar_mass)
{
    // TODO: If I need more derivatives of sound speed, then I might need to compute the effective molar mass inside
    // this function
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    Number cp = calc_cp(eos, c, x, T);
    Number cv = calc_cv(eos, c, x, T);
    Number dp_dc = calc_dp_dc(eos, c, x, T);
    return cp * dp_dc / (effective_molar_mass * cv);
}

/**
 * @brief Chemical potentials @f$\mu_i = \partial A/\partial n_i@f$ of all
 *        components (ideal + residual).
 * @param eos The equation of state.
 * @param rho_i Partial molar concentrations [mol/m^3].
 * @param T   Temperature [K].
 * @param[out] chemical_potential Output chemical potentials [J/mol]. Overwritten (zeroed then filled).
 * @pre `rho_i.size() == eos.size()`
 * @pre `chemical_potential.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_chemical_potential(const EoS<Ideal, Residual>& eos, std::span<const Number, N> rho_i, const Number T,
                             std::span<Number, N> chemical_potential)
{
    assert(rho_i.size() == eos.size());
    assert(chemical_potential.size() == eos.size());
    // Enzyme reverse mode accumulates into the output, so it must start zeroed;
    // the ideal and residual contributions are then summed in place.
    std::fill(chemical_potential.begin(), chemical_potential.end(), Number{0});
    detail::calc_dPsi_drhoi<1>(eos.ideal(), rho_i.data(), T, chemical_potential.data());
    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, chemical_potential.data());
}

/**
 * @brief Forward-mode alternative to calc_chemical_potential.
 *
 * Computes the identical chemical potentials, but obtains each contribution's
 * @f$\partial\Psi/\partial\rho_i@f$ gradient with `eos.size()` Enzyme
 * forward-mode passes (detail::accumulate_dPsi_drhoi_fwd) instead of a single
 * reverse-mode pass. Provided so the two differentiation strategies can be
 * benchmarked against each other.
 *
 * @param eos The equation of state.
 * @param rho_i Partial molar concentrations [mol/m^3].
 * @param T   Temperature [K].
 * @param[out] chemical_potential Output chemical potentials [J/mol]. Overwritten (zeroed then filled).
 * @pre `rho_i.size() == eos.size()`
 * @pre `chemical_potential.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_chemical_potential_fwd(const EoS<Ideal, Residual>& eos, std::span<const Number, N> rho_i, const Number T,
                                 std::span<Number, N> chemical_potential)
{
    assert(rho_i.size() == eos.size());
    assert(chemical_potential.size() == eos.size());
    std::fill(chemical_potential.begin(), chemical_potential.end(), Number{0});
    detail::accumulate_dPsi_drhoi_fwd(eos.ideal(), rho_i, T, chemical_potential.data());
    detail::accumulate_dPsi_drhoi_fwd(eos.residual(), rho_i, T, chemical_potential.data());
}

/**
 * @brief Natural log of the fugacity coefficients,
 *        @f$\ln\varphi_i = \mu_i^{\text{res}}/(RT) - \ln Z@f$.
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @param rho_i Partial molar concentrations [mol/m^3] (should equal `x*c`).
 * @param[out] log_fug_coeff Output @f$\ln\varphi_i@f$ [-] (length `eos.size()`).
 *             Overwritten (zeroed then filled).
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_log_fugacity_coeff(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x,
                             const Number T, const std::span<const Number, N> rho_i, std::span<Number, N> log_fug_coeff)
{
    const Number invT = Number{1} / T;
    const Number Z = Number{1} + detail::calc_lambda<0, 1>(eos.residual(), c, x.data(), invT);
    const Number lnZ = std::log(Z);

    // Enzyme reverse mode accumulates into the output, so zero it first.
    std::fill(log_fug_coeff.begin(), log_fug_coeff.end(), Number{0});
    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, log_fug_coeff.data());
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invRT = Number{1} / (R * T);
    for (auto& ln_phi_i : log_fug_coeff) {
        ln_phi_i *= invRT;
        ln_phi_i -= lnZ;
    }
}

/**
 * @brief Forward-mode alternative to calc_log_fugacity_coeff.
 *
 * Identical result; the residual chemical-potential gradient is obtained with
 * `eos.size()` Enzyme forward-mode passes instead of one reverse-mode pass.
 * Provided for benchmarking the two strategies.
 *
 * @param eos The equation of state.
 * @param c   Molar concentration [mol/m^3].
 * @param x   Mole fractions [-].
 * @param T   Temperature [K].
 * @param rho_i Partial molar concentrations [mol/m^3] (should equal `x*c`).
 * @param[out] log_fug_coeff Output @f$\ln\varphi_i@f$ [-] (length `eos.size()`).
 *             Overwritten (zeroed then filled).
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_log_fugacity_coeff_fwd(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x,
                                 const Number T, const std::span<const Number, N> rho_i,
                                 std::span<Number, N> log_fug_coeff)
{
    const Number invT = Number{1} / T;
    const Number Z = Number{1} + detail::calc_lambda<0, 1>(eos.residual(), c, x.data(), invT);
    const Number lnZ = std::log(Z);

    std::fill(log_fug_coeff.begin(), log_fug_coeff.end(), Number{0});
    detail::accumulate_dPsi_drhoi_fwd(eos.residual(), rho_i, T, log_fug_coeff.data());
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invRT = Number{1} / (R * T);
    for (auto& ln_phi_i : log_fug_coeff) {
        ln_phi_i *= invRT;
        ln_phi_i -= lnZ;
    }
}

/**
 * @brief Fugacities @f$f_i = \rho_i RT\,\exp(\mu_i^{\text{res}}/(RT))@f$.
 * @param eos The equation of state.
 * @param rho_i Partial molar concentrations [mol/m^3] (length `eos.size()`).
 * @param T   Temperature [K].
 * @param[out] fugacity Output fugacities [Pa] (length `eos.size()`).
 *             Overwritten (zeroed then filled).
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_fugacity(const EoS<Ideal, Residual>& eos, std::span<const Number, N> rho_i, const Number T,
                   std::span<Number, N> fugacity)
{
    // Enzyme reverse mode accumulates into the output, so zero it first. After
    // the call `fugacity[idx]` holds the residual chemical potential mu_i^res.
    std::fill(fugacity.begin(), fugacity.end(), Number{0});
    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, fugacity.data());
    constexpr Number R = ideal_gas_constant<Number>;
    const Number RT = R * T;
    const Number invRT = Number{1} / RT;
    for (std::size_t idx = 0; idx < fugacity.size(); ++idx) {
        fugacity[idx] = rho_i[idx] * RT * std::exp(fugacity[idx] * invRT);
    }
}

/**
 * @brief Forward-mode alternative to calc_fugacity.
 *
 * Identical result; the residual chemical-potential gradient is obtained with
 * `eos.size()` Enzyme forward-mode passes instead of one reverse-mode pass.
 * Provided for benchmarking the two strategies.
 *
 * @param eos The equation of state.
 * @param rho_i Partial molar concentrations [mol/m^3] (length `eos.size()`).
 * @param T   Temperature [K].
 * @param[out] fugacity Output fugacities [Pa] (length `eos.size()`).
 *             Overwritten (zeroed then filled).
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_fugacity_fwd(const EoS<Ideal, Residual>& eos, std::span<const Number, N> rho_i, const Number T,
                       std::span<Number, N> fugacity)
{
    std::fill(fugacity.begin(), fugacity.end(), Number{0});
    detail::accumulate_dPsi_drhoi_fwd(eos.residual(), rho_i, T, fugacity.data());
    constexpr Number R = ideal_gas_constant<Number>;
    const Number RT = R * T;
    const Number invRT = Number{1} / RT;
    for (std::size_t idx = 0; idx < fugacity.size(); ++idx) {
        fugacity[idx] = rho_i[idx] * RT * std::exp(fugacity[idx] * invRT);
    }
}

} // namespace glis::eos