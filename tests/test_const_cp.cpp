//
// Unit tests for the constant-cp ideal-gas model (glis::eos::ConstantCp).
//
// Most of the heavy lifting is delegated to the reusable, model-agnostic helpers
// in derivative_test_harness.hpp:
//   - check_helmholtz_consistency       (Psi == c*a, Psi == sum_i Psi_i)
//   - check_ideal_gas_pressure          (p == c R T)
//   - check_euler_pressure              (sum_i rho_i mu_i - Psi == calc_pressure)
//   - run_derivative_consistency_tests  (every property vs. finite differences)
// What remains here is genuinely ConstantCp-specific: the per-species kernel
// algebra and the caloric properties checked against the *physical* reference
// data (h_ref, s_ref, c_p), which the generic helpers cannot know.
//
// Physical references (pure ideal gas, constant isobaric molar heat capacity
// c_p, referenced to a state (T_ref, p_ref)):
//     h(T)    = h_ref + c_p (T - T_ref)
//     s(T, c) = s_ref + (c_p - R) ln(T / T_ref) - R ln(c / c_ref)
//     c_p     = c_p  (constant),   with c_ref = p_ref / (R T_ref).
// At the reference state (T = T_ref, c = c_ref): h = h_ref and s = s_ref.
//
#include "derivative_test_harness.hpp"
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

constexpr std::array<Input<1>, 1> unary_inputs{{{T_ref, p_ref, cp_in, h_ref, s_ref}}};

} // namespace

int main()
{
    suite<"const_cp"> const_cp = [] {
        const double R = ge::ideal_gas_constant<double>;

        // ===================================================================
        // Generic structural consistency (delegated to the shared helper):
        //   Psi == c*a   and   Psi == sum_i Psi_i.
        // ===================================================================
        "helmholtz consistency (generic)"_test = [&] {
            const ge::ConstantCp<2> model(binary_inputs);
            for (const double T : {280.0, 340.0, 410.0}) {
                check_helmholtz_consistency<2>(model, {40.0, 70.0}, T);
                check_helmholtz_consistency<2>(model, {5.0, 95.0}, T);
                check_helmholtz_consistency<2>(model, {120.0, 30.0}, T);
            }
        };

        // ===================================================================
        // Ideal-gas pressure p = c R T (delegated to the shared helper).
        // ===================================================================
        "ideal-gas pressure == c R T"_test = [&] {
            auto eos = make_const_cp_eos<2>(binary_inputs);
            for (const double c : {20.0, 80.0, 200.0}) {
                for (const double T : {270.0, 330.0, 410.0}) {
                    check_ideal_gas_pressure<2>(eos, c, {0.35, 0.65}, T);
                }
            }
        };

        // ===================================================================
        // Pressure via the Euler relation, sum_i rho_i mu_i - Psi == p
        // (delegated to the shared helper). Exercises the multi-component
        // reverse-mode chemical potentials.
        // ===================================================================
        "pressure via Euler relation (mixture)"_test = [&] {
            auto eos = make_const_cp_eos<2>(binary_inputs);
            for (const double c : {30.0, 90.0, 175.0}) {
                for (const double T : {285.0, 360.0}) {
                    check_euler_pressure<2>(eos, {0.4 * c, 0.6 * c}, T);
                }
            }
        };

        // ===================================================================
        // ConstantCp-specific caloric properties vs the PHYSICAL reference data
        // (the generic helpers don't know h_ref/s_ref/c_p). This is the guard
        // that the Helmholtz formula itself is correct.
        // ===================================================================
        "single-species reference state"_test = [&] {
            auto eos = make_const_cp_eos<1>(unary_inputs);
            const double c_ref = p_ref / (R * T_ref);
            const std::array<double, 1> x{1.0};
            const std::span<const double, 1> xs{x};

            check_rel("calc_cp       == c_p", ge::calc_cp(eos, c_ref, xs, T_ref), cp_in, 1e-9);
            check_rel("calc_enthalpy == h_ref", ge::calc_enthalpy(eos, c_ref, xs, T_ref), h_ref, 1e-9);
            check_rel("calc_entropy  == s_ref", ge::calc_entropy(eos, c_ref, xs, T_ref), s_ref, 1e-9);
        };

        "single-species off-reference caloric"_test = [&] {
            auto eos = make_const_cp_eos<1>(unary_inputs);
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
        // Full derivative-consistency sweep: every property in
        // core_calculations.hpp checked against 4th-order finite differences of
        // the model's own Helmholtz energy (delegated to the shared harness).
        // Covers both single-component and mixture states, in debug AND release.
        // ===================================================================
        "derivative consistency vs finite differences"_test = [&] {
            auto unary = make_const_cp_eos<1>(unary_inputs);
            run_derivative_consistency_tests<1>(unary, 120.0, {1.0}, 310.0);
            run_derivative_consistency_tests<1>(unary, 200.0, {1.0}, 360.0);

            auto binary = make_const_cp_eos<2>(binary_inputs);
            run_derivative_consistency_tests<2>(binary, 100.0, {0.4, 0.6}, 300.0);
            run_derivative_consistency_tests<2>(binary, 250.0, {0.7, 0.3}, 350.0);
            run_derivative_consistency_tests<2>(binary, 40.0, {0.5, 0.5}, 280.0);
        };
    };
}
