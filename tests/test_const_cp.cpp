//
// Unit tests for the constant-cp ideal-gas model (glis::eos::ConstantCp).
//
// The model is exercised on its own (per-species kernel algebra) and paired with
// NoResidual so the thermodynamic property routines in core_calculations.hpp can
// be evaluated.
//
// Physical references (pure ideal gas, temperature-independent isobaric molar
// heat capacity c_p, referenced to a state (T_ref, p_ref)):
//     h(T)    = h_ref + c_p (T - T_ref)
//     s(T, c) = s_ref + (c_p - R) ln(T / T_ref) - R ln(c / c_ref)
//     c_p     = c_p  (constant),   with c_ref = p_ref / (R T_ref).
// At the reference state (T = T_ref, c = c_ref): h = h_ref and s = s_ref.
//
// ----------------------------------------------------------------------------
// History (two bugs found via these tests, both now fixed):
//
//  (A) FORMULA -- the molar Helmholtz was missing a -R*T term, so it was offset
//      by +R*T (a Gibbs-like energy). This left p, h, u, c_p, c_v correct but
//      made the entropy come out as s_ref - R. Fixed by subtracting R*T*x_i in
//      calc_helmholtz_i and R*T*rho_i in calc_helmholtz_density_i. The
//      finite-difference caloric test below is the autodiff-independent guard.
//
//  (B) AUTODIFF -- in the debug build (-O1) the Enzyme-differentiated multi-
//      component properties (calc_entropy/c_p/c_v/internal_energy/enthalpy and
//      the chemical-potential / Euler-relation check) disagreed with finite
//      differences, while release (-O3) was correct. Root cause: at -O1 the
//      hoisted pre-calculation struct and per-species parameter struct crossed
//      un-inlined call boundaries as aggregates that Enzyme could not analyse.
//      Fixed by marking the differentiated chain [[gnu::always_inline]] so the
//      whole expression flattens to SSA before Enzyme runs, at any -O level.
//      The mixture Euler-relation test below is the guard (run it in debug!).
// ----------------------------------------------------------------------------
//
#include "derivative_test_harness.hpp" // check_rel, central_diff
#include "eoslab/core/core_calculations.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/ideal_models/const_cp.hpp"
#include "eoslab/residual_models/no_residual.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <span>

using namespace boost::ut;
using namespace eoslab_test;

namespace {

namespace ge = glis::eos;
using ld = long double;

template<std::size_t N> using Input = typename ge::ConstantCp<N>::SpeciesInput;

// Build a complete EoS: a constant-cp ideal contribution + a vanishing residual.
template<std::size_t N> auto make_const_cp_eos(const std::array<Input<N>, N>& in)
{
    return ge::EoS<ge::ConstantCp<N>, ge::NoResidual<N>>{ge::ConstantCp<N>(in), ge::NoResidual<N>{}};
}

// A representative two-component parameter set. The two species deliberately use
// *different* reference temperatures and pressures to exercise per-species
// reference handling.
constexpr std::array<Input<2>, 2> binary_inputs{{
    {/*T_ref*/ 300.0, /*p_ref*/ 1.0e5, /*c_p*/ 29.1, /*h_ref*/ 1500.0, /*s_ref*/ 191.0},
    {/*T_ref*/ 320.0, /*p_ref*/ 9.0e4, /*c_p*/ 33.6, /*h_ref*/ -2200.0, /*s_ref*/ 189.0},
}};

// Single-species reference data for the caloric checks.
constexpr double T_ref = 300.0;
constexpr double p_ref = 1.0e5;
constexpr double cp_in = 29.1;
constexpr double h_ref = -1234.0;
constexpr double s_ref = 205.0;

} // namespace

int main()
{
    suite<"const_cp"> const_cp = [] {
        const double R = ge::ideal_gas_constant<double>;

        // ===================================================================
        // 1. Per-species kernel algebra (no autodiff).
        //    The molar and density kernels must be mutually consistent:
        //        c * a_i = Psi_i           and        c * a = Psi.
        // ===================================================================
        "kernel consistency: c*a_i == Psi_i"_test = [&] {
            const ge::ConstantCp<2> model(binary_inputs);
            for (const double T : {280.0, 340.0, 410.0}) {
                for (const std::array<double, 2> rho : {std::array<double, 2>{40.0, 70.0},
                                                        std::array<double, 2>{5.0, 95.0},
                                                        std::array<double, 2>{120.0, 30.0}}) {
                    const double c = rho[0] + rho[1];
                    const std::array<double, 2> x{rho[0] / c, rho[1] / c};

                    const double a = model.calc_helmholtz(c, x.data(), T);
                    const double psi = model.calc_helmholtz_density(rho.data(), T);
                    check_rel("Psi == c * a", psi, c * a, 1e-11);

                    const auto pre_molar = model.perform_pre_calculations(c, x.data(), T);
                    const auto pre_density = model.perform_pre_calculations(rho.data(), T);
                    std::array<double, 2> psi_partial{};
                    model.calc_partial_helmholtz(rho.data(), T, psi_partial.data());

                    for (std::size_t i = 0; i < 2; ++i) {
                        const double a_i =
                            model.calc_helmholtz_i(c, x.data(), T, i, model.get_parameters(i), pre_molar);
                        const double psi_i =
                            model.calc_helmholtz_density_i(rho.data(), T, i, model.get_parameters(i), pre_density);
                        check_rel("c * a_i == Psi_i", c * a_i, psi_i, 1e-11);
                        check_rel("Psi_i == partial[i]", psi_i, psi_partial[i], 1e-12);
                    }
                }
            }
        };

        // ===================================================================
        // 2. Ideal-gas pressure: p = c R T (mixture, via calc_pressure).
        // ===================================================================
        "ideal-gas pressure == c R T"_test = [&] {
            auto eos = make_const_cp_eos<2>(binary_inputs);
            const std::array<double, 2> x{0.35, 0.65};
            const std::span<const double, 2> xs{x};
            for (const double c : {20.0, 80.0, 200.0}) {
                for (const double T : {270.0, 330.0, 410.0}) {
                    check_rel("p == c R T", ge::calc_pressure(eos, c, xs, T), c * R * T, 1e-12);
                }
            }
        };

        // ===================================================================
        // 3. Reference-state caloric properties from FINITE DIFFERENCES of the
        //    model's own Helmholtz energy (autodiff-independent). This isolates
        //    the FORMULA from the autodiff layer.
        //
        //    Expected at (T_ref, c_ref): h = h_ref, c_p = c_p, s = s_ref.
        //    >>> The entropy check currently FAILS by exactly R (see note (A)).
        // ===================================================================
        "reference caloric via finite differences (formula check)"_test = [&] {
            const std::array<Input<1>, 1> in{{{T_ref, p_ref, cp_in, h_ref, s_ref}}};
            const ge::ConstantCp<1> model(in);
            const ld Rl = ge::ideal_gas_constant<ld>;
            const ld c_ref = p_ref / (Rl * T_ref);
            const std::array<ld, 1> x{1.0L};
            auto a = [&](ld c, ld T) { return model.calc_helmholtz(c, x.data(), T); };

            const ld hT = static_cast<ld>(T_ref) * 1e-4L;
            const ld hc = c_ref * 1e-4L;
            const ld a0 = a(c_ref, T_ref);
            const ld aT = central_diff([&](ld T) { return a(c_ref, T); }, static_cast<ld>(T_ref), hT);
            const ld aTT = central_diff(
                [&](ld T) { return central_diff([&](ld t) { return a(c_ref, t); }, T, hT); }, static_cast<ld>(T_ref), hT);
            const ld ac = central_diff([&](ld c) { return a(c, T_ref); }, c_ref, hc);

            const ld p_fd = c_ref * c_ref * ac;          // p  = c^2 da/dc
            const ld s_fd = -aT;                         // s  = -da/dT
            const ld u_fd = a0 - (static_cast<ld>(T_ref) * aT);
            const ld h_fd = u_fd + (p_fd / c_ref);       // h  = u + p/c
            const ld cv_fd = -static_cast<ld>(T_ref) * aTT;
            const ld cp_fd = cv_fd + Rl;                 // ideal gas: c_p = c_v + R

            check_rel("FD pressure  == c_ref R T_ref", static_cast<double>(p_fd), static_cast<double>(c_ref * Rl * T_ref),
                      1e-7);
            check_rel("FD enthalpy  == h_ref", static_cast<double>(h_fd), h_ref, 1e-7);
            check_rel("FD c_p       == c_p", static_cast<double>(cp_fd), cp_in, 1e-6);
            check_rel("FD entropy   == s_ref", static_cast<double>(s_fd), s_ref, 1e-7);
        };

        // ===================================================================
        // 4. Reference-state caloric via the library API (core_calculations.hpp)
        //    for a single species (autodiff is reliable for N == 1).
        //
        //    >>> The entropy check currently FAILS by exactly R (note (A)).
        // ===================================================================
        "single-species reference state (library API)"_test = [&] {
            const std::array<Input<1>, 1> in{{{T_ref, p_ref, cp_in, h_ref, s_ref}}};
            auto eos = make_const_cp_eos<1>(in);
            const double c_ref = p_ref / (R * T_ref);
            const std::array<double, 1> x{1.0};
            const std::span<const double, 1> xs{x};

            check_rel("calc_cp       == c_p", ge::calc_cp(eos, c_ref, xs, T_ref), cp_in, 1e-9);
            check_rel("calc_enthalpy == h_ref", ge::calc_enthalpy(eos, c_ref, xs, T_ref), h_ref, 1e-9);
            check_rel("calc_entropy  == s_ref", ge::calc_entropy(eos, c_ref, xs, T_ref), s_ref, 1e-9);
        };

        "single-species off-reference caloric (library API)"_test = [&] {
            const std::array<Input<1>, 1> in{{{T_ref, p_ref, cp_in, h_ref, s_ref}}};
            auto eos = make_const_cp_eos<1>(in);
            const double c_ref = p_ref / (R * T_ref);
            const std::array<double, 1> x{1.0};
            const std::span<const double, 1> xs{x};

            for (const double T : {275.0, 350.0, 425.0}) {
                for (const double c : {c_ref, 0.5 * c_ref, 3.0 * c_ref}) {
                    const double h_expected = h_ref + (cp_in * (T - T_ref));
                    const double s_expected = s_ref + ((cp_in - R) * std::log(T / T_ref)) - (R * std::log(c / c_ref));
                    check_rel("calc_enthalpy(T)", ge::calc_enthalpy(eos, c, xs, T), h_expected, 1e-9);
                    check_rel("calc_cp(T,c)", ge::calc_cp(eos, c, xs, T), cp_in, 1e-9);
                    check_rel("calc_entropy(T,c)", ge::calc_entropy(eos, c, xs, T), s_expected, 1e-9);
                }
            }
        };

        // ===================================================================
        // 5. Pressure via the Euler relation (mixtures):
        //        p = sum_i rho_i mu_i - Psi
        //    with mu_i the chemical potentials (reverse-mode autodiff of Psi).
        //    For this model the identity reduces analytically to p = c R T, so
        //    this also cross-checks calc_pressure.
        //
        //    >>> Currently FAILS for mixtures due to the autodiff issue (B);
        //        the chemical potentials returned for N > 1 are inconsistent.
        // ===================================================================
        "pressure via Euler relation (mixture)"_test = [&] {
            auto eos = make_const_cp_eos<2>(binary_inputs);
            const std::array<double, 2> x{0.4, 0.6};
            const std::span<const double, 2> xs{x};

            for (const double c : {30.0, 90.0, 175.0}) {
                for (const double T : {285.0, 360.0}) {
                    std::array<double, 2> rho{x[0] * c, x[1] * c};
                    const std::span<const double, 2> rhos{rho};

                    std::array<double, 2> mu{};
                    ge::calc_chemical_potential(eos, rhos, T, std::span<double, 2>{mu});

                    const double psi = eos.ideal().calc_helmholtz_density(rho.data(), T) +
                                       eos.residual().calc_helmholtz_density(rho.data(), T);
                    const double p_euler = (rho[0] * mu[0]) + (rho[1] * mu[1]) - psi;

                    check_rel("Euler p == calc_pressure", p_euler, ge::calc_pressure(eos, c, xs, T), 1e-8);
                    check_rel("Euler p == c R T", p_euler, c * R * T, 1e-8);
                }
            }
        };
    };
}
