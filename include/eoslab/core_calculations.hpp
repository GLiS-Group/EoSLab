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
#include "eoslab/concepts.hpp"
#include "eoslab/eos_pair.hpp"
#include "eoslab/numbers.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>

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
 * Calls the model's `calc_partial_helmholtz` to fill @p Psi_i and returns their
 * sum. This is the scalar that detail::calc_dPsi_drhoi differentiates (reverse
 * mode) to obtain chemical potentials.
 *
 * @param  eos   A single-contribution model (ideal or residual).
 * @param  rho_i Partial molar concentrations [mol/m^3].
 * @param  T     Temperature [K].
 * @param  Psi_i Output scratch: per-component Helmholtz energy density [J/m^3].
 * @return Total Helmholtz energy density @f$\Psi@f$ [J/m^3].
 */
template<EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_Psi(const EoS& eos, const Number* GLIS_EOS_RESTRICT rho_i, const Number T,
                              Number* GLIS_EOS_RESTRICT Psi_i)
{
    eos.calc_partial_helmholtz(rho_i, T, Psi_i);
    Number Psi{0};
    for (std::size_t idx = 0; idx < eos.size(); ++idx) {
        Psi += Psi_i[idx];
    }
    return Psi;
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
 * @param  scratch Scratch buffer of length `eos.size()` [J/m^3].
 * @return @f$\partial^{i}\Psi/\partial T^{i}@f$ [J/(m^3 K^i)].
 */
template<int i, EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_dPsi_dT(const EoS& eos, const Number* GLIS_EOS_RESTRICT rho_i, const Number T,
                                  Number* GLIS_EOS_RESTRICT scratch)
{
    static_assert(i >= 0, "The template parameter `i` must be non-negative. It represent the number of derivatives "
                          "with respect to `T`.");
    if constexpr (i == 0) {
        return calc_Psi(eos, rho_i, T, scratch);
    }
    else {
        Number dT{1};
        return __enzyme_fwddiff<Number>((void*)calc_dPsi_dT<i - 1, EoS, Number>, enzyme_const, &eos, enzyme_const,
                                        rho_i, enzyme_dup, T, dT, enzyme_const, scratch);
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
 * @note Neither scratch buffer needs to be initialised by the caller.
 *       @p scratch is a pure output of detail::calc_Psi (fully overwritten by
 *       `calc_partial_helmholtz` before it is read; its contents on entry --
 *       even NaN/Inf -- do not affect the result). @p dscratch (the Enzyme
 *       shadow) MUST start at zero for the reverse pass, so it is zeroed
 *       internally on every call.
 *
 * @tparam i        Derivative order; must be 1.
 * @param  eos      A single-contribution model (ideal or residual).
 * @param  rho_i    Partial molar concentrations [mol/m^3].
 * @param  T        Temperature [K].
 * @param  dPsi_drho Output gradient (length `eos.size()`), accumulated [J/mol].
 * @param  scratch  Caller-allocated scratch of length `eos.size()` [J/m^3];
 *                  receives the per-component Helmholtz density from
 *                  detail::calc_Psi. Contents on entry are irrelevant.
 * @param  dscratch Caller-allocated scratch of length `eos.size()`; used as the
 *                  Enzyme shadow of @p scratch. Contents on entry are irrelevant
 *                  (zeroed internally).
 */
template<int i, EquationOfState EoS, std::floating_point Number>
void calc_dPsi_drhoi(const EoS& eos, const Number* GLIS_EOS_RESTRICT rho_i, const Number T,
                     Number* GLIS_EOS_RESTRICT dPsi_drho, Number* GLIS_EOS_RESTRICT scratch,
                     Number* GLIS_EOS_RESTRICT dscratch)
{
    // TODO: only implemented for 1 derivative because higher order would require tensors!
    static_assert(i > 0, "The template parameter `i` must be positive. It represents the number of derivatives with "
                         "respect to each `rho_i`. Use `calc_Psi()` for the '0th' derivative.");
    // NOTE: Enzyme reverse mode ACCUMULATES into `dPsi_drho`, so the caller must
    // zero it before the first call (and may sum several contributions by
    // calling repeatedly with the same buffer).
    if constexpr (i == 1) {
        // `scratch` holds the per-component Helmholtz density (`Psi_i`) written
        // inside calc_Psi(); because those intermediate values depend on the
        // active inputs `rho_i`, the buffer is an *active* argument and must be
        // shadowed (enzyme_dup) rather than marked const -- otherwise Enzyme
        // drops the dependence through the buffer and returns a zero gradient.
        //
        // The shadow `dscratch` MUST start zeroed: the reverse pass accumulates
        // into it, and a non-zero (or NaN/Inf) value here corrupts the gradient
        // (verified -- garbage produces ~1e303 / NaN chemical potentials). The
        // primal `scratch` needs no initialisation: calc_partial_helmholtz
        // overwrites it before it is read. Both buffers are caller-provided to
        // avoid a per-call allocation.
        std::fill(dscratch, dscratch + eos.size(), Number{0});
        __enzyme_autodiff<Number>((void*)calc_Psi<EoS, Number>, enzyme_const, &eos, enzyme_dup, rho_i, dPsi_drho,
                                  enzyme_const, T, enzyme_dup, scratch, dscratch);
        return;
    }
    else {
        static_assert(false, "Higher order derivatives are not implemented (yet?) because they require tensors!");
        return;
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
 * @param scratch1 Caller-allocated scratch buffer of length `eos.size()` [J/m^3].
 * @param scratch2 Caller-allocated scratch buffer of length `eos.size()` (Enzyme shadow).
 * @note Neither @p scratch1 nor @p scratch2 needs to be initialised by the
 *       caller; both are overwritten/zeroed internally, so you do not need to
 *       preprocess them before calling this function.
 * @pre `rho_i.size() == eos.size()`
 * @pre `chemical_potential.size() == eos.size()`
 * @pre `scratch1.size() == eos.size()`
 * @pre `scratch2.size() == eos.size()`
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_chemical_potential(const EoS<Ideal, Residual>& eos, std::span<const Number, N> rho_i, const Number T,
                             std::span<Number, N> chemical_potential, std::span<Number, N> scratch1,
                             std::span<Number, N> scratch2)
{
    assert(rho_i.size() == eos.size());
    assert(chemical_potential.size() == eos.size());
    assert(scratch1.size() == eos.size());
    assert(scratch2.size() == eos.size());
    // Enzyme reverse mode accumulates into the output, so it must start zeroed;
    // the ideal and residual contributions are then summed in place.
    std::fill(chemical_potential.begin(), chemical_potential.end(), Number{0});
    detail::calc_dPsi_drhoi<1>(eos.ideal(), rho_i.data(), T, chemical_potential.data(), scratch1.data(),
                               scratch2.data());
    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, chemical_potential.data(), scratch1.data(),
                               scratch2.data());
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
 * @param scratch1 Caller-allocated scratch buffer of length `eos.size()` [J/m^3].
 * @param scratch2 Caller-allocated scratch buffer of length `eos.size()` (Enzyme shadow).
 * @note Neither @p scratch1 nor @p scratch2 needs to be initialised by the
 *       caller; both are overwritten/zeroed internally, so you do not need to
 *       preprocess them before calling this function.
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_log_fugacity_coeff(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x,
                             const Number T, const std::span<const Number, N> rho_i, std::span<Number, N> log_fug_coeff,
                             std::span<Number, N> scratch1, std::span<Number, N> scratch2)
{
    // TODO: assertions on span
    // FIXME: In this whole file, if N != std::dynamic_extent, then I can do the size checks at compile time!
    const Number invT = Number{1} / T;
    const Number Z = Number{1} + detail::calc_lambda<0, 1>(eos.residual(), c, x.data(), invT);
    const Number lnZ = std::log(Z);

    // Enzyme reverse mode accumulates into the output, so zero it first.
    std::fill(log_fug_coeff.begin(), log_fug_coeff.end(), Number{0});
    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, log_fug_coeff.data(), scratch1.data(), scratch2.data());
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
 * @param scratch1 Caller-allocated scratch buffer of length `eos.size()` [J/m^3].
 * @param scratch2 Caller-allocated scratch buffer of length `eos.size()` (Enzyme shadow).
 * @note Neither @p scratch1 nor @p scratch2 needs to be initialised by the
 *       caller; both are overwritten/zeroed internally, so you do not need to
 *       preprocess them before calling this function.
 */
template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_fugacity(const EoS<Ideal, Residual>& eos, std::span<const Number, N> rho_i, const Number T,
                   std::span<Number, N> fugacity, std::span<Number, N> scratch1, std::span<Number, N> scratch2)
{
    // Enzyme reverse mode accumulates into the output, so zero it first. After
    // the call `fugacity[idx]` holds the residual chemical potential mu_i^res.
    std::fill(fugacity.begin(), fugacity.end(), Number{0});
    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, fugacity.data(), scratch1.data(), scratch2.data());
    constexpr Number R = ideal_gas_constant<Number>;
    const Number RT = R * T;
    const Number invRT = Number{1} / RT;
    for (std::size_t idx = 0; idx < fugacity.size(); ++idx) {
        fugacity[idx] = rho_i[idx] * RT * std::exp(fugacity[idx] * invRT);
    }
}

} // namespace glis::eos