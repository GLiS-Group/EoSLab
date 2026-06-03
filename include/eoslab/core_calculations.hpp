#pragma once
#include "eoslab/concepts.hpp"
#include "eoslab/eos_pair.hpp"
#include "eoslab/numbers.hpp"

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>

// Enzyme autodiff requires a few global definitions
// TODO: Make these `const` to please clangd?? Try making them const after writing tests to see.
inline int enzyme_dup;
inline int enzyme_dupnoneed;
inline int enzyme_out;
inline int enzyme_const;

// NOLINTBEGIN
template<typename return_type, typename... T> return_type __enzyme_fwddiff(void*, T...);

template<typename return_type, typename... T> return_type __enzyme_autodiff(void*, T...);
// NOLINTEND

namespace glis::eos {

namespace detail {
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
    else if constexpr (N % 2 == 2) {
        Number half = fast_pow<Number, N / 2>(base);
        return half * half;
    }
    else {
        return base * fast_pow<Number, N - 1>(base);
    }
}

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
        return eos.calc_total_helmholtz(c, x, T) / (R * T);
    }
    else if constexpr (i > j) {
        Number dinvT{1.};
        return __enzyme_fwddiff<Number>((void*)calc_alpha<i - 1, j, EoS, Number>, enzyme_const, eos, enzyme_const, c,
                                        enzyme_const, x, enzyme_dup, invT, dinvT);
    }
    else {
        Number dc{1.};
        return __enzyme_fwddiff<Number>((void*)calc_alpha<i, j - 1, EoS, Number>, enzyme_const, eos, enzyme_dup, c, dc,
                                        enzyme_const, x, enzyme_const, invT);
    }
}

template<int i, int j, EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_lambda(const EoS& eos, const Number c, const Number* x, const Number invT)
{
    // TODO: better assertions?
    static_assert(i >= 0, "The template parameter `i` must be non-negative. It represent the number of derivatives "
                          "with respect to `invT`.");
    static_assert(
        j >= 0,
        "The template parameter `j` must be non-negative. It represent the number of derivatives with respect to `c`.");
    return fast_pow<Number, i>(invT) * fast_pow<Number, i>(c) * calc_alpha<i, j, EoS, Number>(eos, c, x, invT);
}

template<EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_Psi(const EoS& eos, const GLIS_EOS_RESTRICT Number* rho_i, const Number T, GLIS_EOS_RESTRICT Number* Psi_i){
    eos.calc_partial_helmholtz(rho_i,  T,  Psi_i);
    Number Psi{0};
    for (std::size_t idx = 0; idx < eos.size(); ++idx){
        Psi += Psi_i[idx];
    };
}

template<int i, EquationOfState EoS, std::floating_point Number>
[[nodiscard]] Number calc_dPsi_dT(const EoS& eos, const GLIS_EOS_RESTRICT Number* rho_i, const Number T, GLIS_EOS_RESTRICT Number* scratch){
    static_assert(i >= 0, "The template parameter `i` must be non-negative. It represent the number of derivatives "
                          "with respect to `T`.");
    if constexpr (i == 0){
        return calc_Psi(eos, rho_i, T, scratch);
    } else {
        Number dT{1};
        return __enzyme_fwddiff<Number>((void*)calc_dPsi_dT<i-1,EoS,Number>,enzyme_const, eos, enzyme_const, rho_i, enzyme_dup, T, dT, enzyme_const, scratch);
    }
}

template<int i, EquationOfState EoS, std::floating_point Number>
void calc_dPsi_drhoi(const EoS& eos, const GLIS_EOS_RESTRICT Number* rho_i, const Number T, GLIS_EOS_RESTRICT Number* dPsi_drho, GLIS_EOS_RESTRICT Number* scratch){
    // TODO: only implemented for 1 derivative because higher order would require tensors!
    static_assert(i>0,"The template parameter `i` must be positive. It represents the number of derivatives with respect to each `rho_i`. Use `calc_Psi()` for the '0th' derivative.");
    // TODO: This ADDS to dPsi_drho, so ensure it is properly initialized (e.g. zero'd out)
    if constexpr (i==1){
        __enzyme_autodiff<Number>(
            (void*)calc_Psi<EoS,Number>,
            enzyme_const, eos,
            enzyme_dup, rho_i, dPsi_drho,
            enzyme_const, T,
            enzyme_const, scratch
        );
        return;
    } else {
        static_assert(false,"Higher order derivatives are not implemented (yet?) because they require tensors!");
        return;
    }
}
} // namespace detail

template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_helmholtz(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    return ideal.calc_helmholtz(c, x.data(), T) + residual.calc_helmholtz(c, x.data(), T);
}

template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_pressure(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return c * R * T * (Number{1} + detail::calc_lambda<0, 1>(residual, c, x, invT));
}

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
    return R * T * (detail::calc_lambda<1, 0>(ideal, c, x, invT) + detail::calc_lambda<1, 0>(residual, c, x, invT));
}

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
           (Number{1} + detail::calc_lambda<0, 1>(residual, c, x, invT) + detail::calc_lambda<1, 0>(ideal, c, x, invT) +
            detail::calc_lambda<1, 0>(residual, c, x, invT));
}

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
    return R * (detail::calc_lambda<1, 0>(ideal, c, x, invT) + detail::calc_lambda<1, 0>(residual, c, x, invT) -
                detail::calc_lambda<0, 0>(ideal, c, x, invT) - detail::calc_lambda<0, 0>(residual, c, x, invT));
}

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
           (Number{1} + detail::calc_lambda<0, 1>(residual, c, x, invT) + detail::calc_lambda<0, 0>(ideal, c, x, invT) +
            detail::calc_lambda<0, 0>(residual, c, x, invT));
}

template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_dp_dc(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * T *
           (Number{1} + (Number{2} * detail::calc_lambda<0, 1>(residual, c, x, invT)) +
            detail::calc_lambda<0, 2>(residual, c, x, invT));
}

template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_dp_dT(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T)
{
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    const Ideal& ideal = eos.ideal();
    const Residual& residual = eos.residual();
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invT = Number{1} / T;
    return R * c *
           (Number{1} + detail::calc_lambda<0, 1>(residual, c, x, invT) -
            detail::calc_lambda<1, 1>(residual, c, x, invT));
}

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
    return -R * (detail::calc_lambda<2, 0>(ideal, c, x, invT) + detail::calc_lambda<2, 0>(residual, c, x, invT));
}

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
    const Number lr_01 = detail::calc_lambda<0, 1>(residual, c, x, invT);
    return R * (-detail::calc_lambda<2, 0>(ideal, c, x, invT) - detail::calc_lambda<2, 0>(residual, c, x, invT) +
                (detail::fast_pow<Number, 2>(Number{1} + lr_01 - detail::calc_lambda<1, 1>(residual, c, x, invT)) /
                    (Number{1} + (Number{2} * lr_01) + detail::calc_lambda<0, 2>(residual, c, x, invT))));
}

template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
Number calc_sound_speed(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T, const Number effective_molar_mass)
{
    // TODO: If I need more derivatives of sound speed, then I might need to compute the effective molar mass inside this function
    // TODO: Consider a custom assertion with a better error message
    assert(x.size() == eos.size());
    assert(T > Number{0});
    Number cp = calc_cp(eos,c,x,T);
    Number cv = calc_cv(eos,x,T);
    Number dp_dc = calc_dp_dc(eos,  c,  x,  T);
    return cp*dp_dc/(effective_molar_mass*cv);
}

template <IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_chemical_potential(const EoS<Ideal, Residual>& eos, std::span<const Number, N> rho_i, const Number T, std::span<Number, N> chemical_potential, std::span<Number, N> scratch){
    // TODO: assertions
    // values accumulate in Enzyme backwards mode
    detail::calc_dPsi_drhoi<1>(eos.ideal(),rho_i.data(), T, chemical_potential.data(), scratch.data());
    detail::calc_dPsi_drhoi<1>(eos.residual(),rho_i.data(), T, chemical_potential.data(), scratch.data());
}

template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_log_fugacity_coeff(const EoS<Ideal, Residual>& eos, const Number c, std::span<const Number, N> x, const Number T, const std::span<const Number, N> rho_i, std::span<Number,N> log_fug_coeff, std::span<Number,N> scratch){
    // TODO: assertions on span
    // FIXME: In this whole file, if N != std::dynamic_extent, then I can do the size checks at compile time!
    const Number invT = Number{1}/ T;
    const Number Z = Number{1} + detail::calc_lambda<0, 1>(eos.residual(), c, x.data(), invT);
    const Number lnZ = std::log(Z);

    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, log_fug_coeff.data(), scratch.data());
    constexpr Number R = ideal_gas_constant<Number>;
    const Number invRT = Number{1} / (R*T);
    for (auto& ln_phi_i : log_fug_coeff){
        ln_phi_i *= invRT;
        ln_phi_i -= lnZ;
    }
}

template<IdealEoS Ideal, ResidualEoS Residual, std::floating_point Number, std::size_t N>
void calc_fugacity(const EoS<Ideal, Residual> & eos, std::span<const Number, N> rho_i, const Number T, std::span<Number,N> fugacity, std::span<Number, N> scratch){
    detail::calc_dPsi_drhoi<1>(eos.residual(), rho_i.data(), T, fugacity.data(), scratch.data());
    constexpr Number R = ideal_gas_constant<Number>;
    const Number RT = R*T;
    for (std::size_t idx = 0; idx < fugacity.size(); ++idx){
        fugacity[idx] = rho_i[idx] * RT * std::exp(fugacity[idx]);
    }
}

} // namespace glis::eos