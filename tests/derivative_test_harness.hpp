#pragma once
//
// Reusable, templated derivative-consistency harness.
//
// Given *any* EoS pair (an instance of glis::eos::EoS<Ideal, Residual>) and a
// thermodynamic state, this checks every derivative-based quantity in
// core_calculations.hpp against an independent finite-difference reference.
//
// The references are built from textbook thermodynamic identities applied to
// the model's own fundamental Helmholtz functions, evaluated in `long double`
// with a 4th-order central finite-difference stencil:
//
//     f'(x) = (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12 h)
//
// Because the references do NOT reuse the library's internal reduced-derivative
// (alpha / lambda) machinery, a passing run validates both the Enzyme autodiff
// AND the closed-form thermodynamic formulas wired on top of it.
//
// To add derivative tests for a newly implemented EoS, simply call
// `run_derivative_consistency_tests(eos, c, x, T)` from a test case.
//
// Units (see eos_test_models.hpp / the library headers):
//   c   [mol/m^3], x [-], T [K], pressure [Pa], energies [J/mol],
//   entropy / heat capacities [J/mol/K], chemical potential [J/mol].
//
#include "eoslab/core/core_calculations.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <string_view>

namespace eoslab_test {

// 4th-order central first derivative of a unary callable, in long double.
[[nodiscard]] long double central_diff(auto f, long double x, long double h)
{
    return (-f(x + (2 * h)) + (8 * f(x + h)) - (8 * f(x - h)) + f(x - (2 * h))) / (12 * h);
}

// Relative-error expectation with an informative message on failure.
inline void check_rel(std::string_view name, double actual, double expected, double reltol)
{
    using namespace boost::ut;
    const double denom = std::max(1.0, std::abs(expected));
    const double rel = std::abs(actual - expected) / denom;
    expect(rel <= reltol) << std::format("{}: actual={:.12g} expected={:.12g} rel_err={:.3e} (tol={:.1e})", name,
                                         actual, expected, rel, reltol);
}

// ---------------------------------------------------------------------------
// Main harness.
//   eos : an EoS<Ideal, Residual>
//   c   : molar concentration                 [mol/m^3]
//   x   : mole fractions (must sum to 1)       [-]
//   T   : temperature                          [K]
//   effective_molar_mass : used only for the speed-of-sound check [kg/mol]
// ---------------------------------------------------------------------------
template<std::size_t N, class EoSPair>
void run_derivative_consistency_tests(const EoSPair& eos, double c, std::array<double, N> x, double T,
                                      double effective_molar_mass = 0.02)
{
    namespace ge = glis::eos;
    using ld = long double;

    const ld R = ge::ideal_gas_constant<ld>;
    const ld c0 = c;
    const ld T0 = T;
    std::array<ld, N> xl{};
    for (std::size_t i = 0; i < N; ++i) {
        xl[i] = static_cast<ld>(x[i]);
    }

    const auto& ideal = eos.ideal();
    const auto& residual = eos.residual();

    // --- fundamental Helmholtz functions in long double -------------------
    auto a_res = [&](ld cc, ld TT) { return residual.calc_helmholtz(cc, xl.data(), TT); };
    auto a_tot = [&](ld cc, ld TT) {
        return ideal.calc_helmholtz(cc, xl.data(), TT) + residual.calc_helmholtz(cc, xl.data(), TT);
    };

    const ld hc = c0 * 1e-3L;
    const ld hT = T0 * 1e-3L;

    // residual c-derivatives (at fixed T0)
    const ld a_res_c = central_diff([&](ld cc) { return a_res(cc, T0); }, c0, hc);
    const ld a_res_cc =
        central_diff([&](ld cc) { return central_diff([&](ld c2) { return a_res(c2, T0); }, cc, hc); }, c0, hc);
    // residual mixed c-T derivative: d/dT ( da_res/dc )
    const ld a_res_cT =
        central_diff([&](ld TT) { return central_diff([&](ld cc) { return a_res(cc, TT); }, c0, hc); }, T0, hT);

    // total T-derivatives (at fixed c0)
    const ld a_tot_0 = a_tot(c0, T0);
    const ld a_tot_T = central_diff([&](ld TT) { return a_tot(c0, TT); }, T0, hT);
    const ld a_tot_TT =
        central_diff([&](ld TT) { return central_diff([&](ld T2) { return a_tot(c0, T2); }, TT, hT); }, T0, hT);

    // --- reference thermodynamic quantities -------------------------------
    const ld p_ref = (c0 * R * T0) + (c0 * c0 * a_res_c);
    const ld dp_dc_ref = (R * T0) + (2 * c0 * a_res_c) + (c0 * c0 * a_res_cc);
    const ld dp_dT_ref = (c0 * R) + (c0 * c0 * a_res_cT);
    const ld u_ref = a_tot_0 - (T0 * a_tot_T);
    const ld s_ref = -a_tot_T;
    const ld cv_ref = -T0 * a_tot_TT;
    const ld h_ref = u_ref + (p_ref / c0);
    const ld g_ref = a_tot_0 + (p_ref / c0);
    const ld cp_ref = cv_ref + (T0 * dp_dT_ref * dp_dT_ref / (c0 * c0 * dp_dc_ref));
    const ld w2_ref = cp_ref * dp_dc_ref / (static_cast<ld>(effective_molar_mass) * cv_ref);

    // --- framework (double, Enzyme) values --------------------------------
    std::array<double, N> xarr = x;
    std::span<const double, N> xs{xarr};

    check_rel("helmholtz", ge::calc_helmholtz(eos, c, xs, T), static_cast<double>(a_tot_0), 1e-12);
    check_rel("pressure", ge::calc_pressure(eos, c, xs, T), static_cast<double>(p_ref), 1e-8);
    check_rel("internal_energy", ge::calc_internal_energy(eos, c, xs, T), static_cast<double>(u_ref), 1e-8);
    check_rel("enthalpy", ge::calc_enthalpy(eos, c, xs, T), static_cast<double>(h_ref), 1e-8);
    check_rel("entropy", ge::calc_entropy(eos, c, xs, T), static_cast<double>(s_ref), 1e-8);
    check_rel("gibbs", ge::calc_gibbs(eos, c, xs, T), static_cast<double>(g_ref), 1e-8);
    check_rel("dp_dc", ge::calc_dp_dc(eos, c, xs, T), static_cast<double>(dp_dc_ref), 1e-6);
    check_rel("dp_dT", ge::calc_dp_dT(eos, c, xs, T), static_cast<double>(dp_dT_ref), 1e-6);
    check_rel("cv", ge::calc_cv(eos, c, xs, T), static_cast<double>(cv_ref), 1e-6);
    check_rel("cp", ge::calc_cp(eos, c, xs, T), static_cast<double>(cp_ref), 1e-6);
    check_rel("sound_speed_squared", ge::calc_sound_speed_squared(eos, c, xs, T, effective_molar_mass),
              static_cast<double>(w2_ref), 1e-6);

    // --- partial-molar quantities (reverse-mode autodiff) -----------------
    // State in partial concentrations: rho_i = x_i * c.
    std::array<ld, N> rhol{};
    std::array<double, N> rho{};
    for (std::size_t i = 0; i < N; ++i) {
        rhol[i] = xl[i] * c0;
        rho[i] = x[i] * c;
    }

    auto Psi_tot = [&](std::array<ld, N> r) {
        std::array<ld, N> oi{};
        std::array<ld, N> orr{};
        ideal.calc_partial_helmholtz(r.data(), T0, oi.data());
        residual.calc_partial_helmholtz(r.data(), T0, orr.data());
        ld s{0};
        for (std::size_t i = 0; i < N; ++i) {
            s += oi[i] + orr[i];
        }
        return s;
    };
    auto Psi_res = [&](std::array<ld, N> r) {
        std::array<ld, N> orr{};
        residual.calc_partial_helmholtz(r.data(), T0, orr.data());
        ld s{0};
        for (std::size_t i = 0; i < N; ++i) {
            s += orr[i];
        }
        return s;
    };

    // mu_i = d Psi_tot / d rho_i ; mu_i^res = d Psi_res / d rho_i
    std::array<ld, N> mu_ref{};
    std::array<ld, N> mu_res_ref{};
    for (std::size_t i = 0; i < N; ++i) {
        const ld hr = rhol[i] * 1e-3L;
        mu_ref[i] = central_diff(
            [&](ld v) {
                auto r = rhol;
                r[i] = v;
                return Psi_tot(r);
            },
            rhol[i], hr);
        mu_res_ref[i] = central_diff(
            [&](ld v) {
                auto r = rhol;
                r[i] = v;
                return Psi_res(r);
            },
            rhol[i], hr);
    }

    std::span<const double, N> rhos{rho};
    std::array<double, N> chem{};
    std::array<double, N> scratch1{};
    std::array<double, N> scratch2{};
    ge::calc_chemical_potential(eos, rhos, T, std::span<double, N>{chem}, std::span<double, N>{scratch1},
                                std::span<double, N>{scratch2});
    for (std::size_t i = 0; i < N; ++i) {
        check_rel(std::format("chemical_potential[{}]", i), chem[i], static_cast<double>(mu_ref[i]), 1e-7);
    }

    // Z = p / (c R T)
    const ld Z_ref = p_ref / (c0 * R * T0);
    std::array<double, N> logphi{};
    ge::calc_log_fugacity_coeff(eos, c, xs, T, rhos, std::span<double, N>{logphi}, std::span<double, N>{scratch1},
                                std::span<double, N>{scratch2});
    for (std::size_t i = 0; i < N; ++i) {
        const ld ref = (mu_res_ref[i] / (R * T0)) - std::log(Z_ref);
        check_rel(std::format("log_fugacity_coeff[{}]", i), logphi[i], static_cast<double>(ref), 1e-7);
    }

    std::array<double, N> fug{};
    ge::calc_fugacity(eos, rhos, T, std::span<double, N>{fug}, std::span<double, N>{scratch1},
                      std::span<double, N>{scratch2});
    for (std::size_t i = 0; i < N; ++i) {
        const ld ref = rhol[i] * R * T0 * std::exp(mu_res_ref[i] / (R * T0));
        check_rel(std::format("fugacity[{}]", i), fug[i], static_cast<double>(ref), 1e-7);
    }
}

} // namespace eoslab_test
