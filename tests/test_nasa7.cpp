//
// Unit tests for the NASA-7 polynomial ideal-gas model (glis::eos::Nasa7).
//
// As in test_const_cp.cpp, the model-agnostic structural and derivative checks
// are delegated to the reusable helpers in derivative_test_harness.hpp:
//   - check_helmholtz_consistency       (Psi == c*a, Psi == sum_i Psi_i)
//   - check_ideal_gas_pressure          (p == c R T)
//   - check_euler_pressure              (sum_i rho_i mu_i - Psi == calc_pressure)
//   - run_derivative_consistency_tests  (every property vs. finite differences)
// What remains here is genuinely NASA7-specific: the per-species kernel algebra
// and the caloric properties checked against the *physical* NASA-7 relations,
// which the generic helpers cannot know.
//
// NASA-7 standard-state relations for a single species (T in [T_low, T_high]),
// with R the molar gas constant:
//
//   c_p(T) / R       = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4
//   h(T)  / (R T)    = a0 + a1/2 T + a2/3 T^2 + a3/4 T^3 + a4/5 T^4 + a5/T
//   s(T)  / R        = a0 ln(T) + a1 T + a2/2 T^2 + a3/3 T^3 + a4/4 T^4 + a6
//
// The standard-state entropy s(T) above is the value at the reference
// concentration c_ref = p_ref / (R T_ref); at a general concentration c the
// model entropy carries the ideal-gas mixing term  -R ln(c / c_ref).
// Enthalpy and c_p are pressure/concentration independent for an ideal gas.
//
#include "derivative_test_harness.hpp"
#include "eoslab/core/core_calculations.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/ideal_models/nasa7.hpp"
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

template<std::size_t N> using Input = typename ge::Nasa7<N>::SpeciesInput;

// Build a complete EoS: a NASA-7 ideal contribution + a vanishing residual.
template<std::size_t N> auto make_nasa7_eos(const std::array<Input<N>, N>& in)
{
    return ge::EoS<ge::Nasa7<N>, ge::NoResidual<N>>{ge::Nasa7<N>(in), ge::NoResidual<N>{}};
}

// Representative low-temperature (200-1000 K) NASA-7 coefficient sets for N2 and
// O2. The two species deliberately use *different* reference temperatures and
// pressures to exercise per-species reference handling.
constexpr std::array<Input<2>, 2> binary_inputs{{
    {/*a0*/ .a0=3.53100528, /*a1*/ .a1=-1.23660988e-4, /*a2*/ .a2=-5.02999433e-7, /*a3*/ .a3=2.43530612e-9, /*a4*/ .a4=-1.40881235e-12,
     /*a5*/ .a5=-1046.97628, /*a6*/ .a6=2.96747038, /*T_ref*/ .T_ref=300.0, /*p_ref*/ .p_ref=1.0e5},
    {/*a0*/ .a0=3.78245636, /*a1*/ .a1=-2.99673416e-3, /*a2*/ .a2=9.84730201e-6, /*a3*/ .a3=-9.68129509e-9, /*a4*/ .a4=3.24372837e-12,
     /*a5*/ .a5=-1063.94356, /*a6*/ .a6=3.65767573, /*T_ref*/ .T_ref=320.0, /*p_ref*/ .p_ref=9.0e4},
}};

// Single-species reference data (N2) for the caloric checks.
constexpr double T_ref = 300.0;
constexpr double p_ref = 1.0e5;

constexpr std::array<Input<1>, 1> unary_inputs{{{/*a0*/ .a0=3.53100528, /*a1*/ .a1=-1.23660988e-4, /*a2*/ .a2=-5.02999433e-7,
                                                 /*a3*/ .a3=2.43530612e-9, /*a4*/ .a4=-1.40881235e-12, /*a5*/ .a5=-1046.97628,
                                                 /*a6*/ .a6=2.96747038, .T_ref=T_ref, .p_ref=p_ref}}};

// --- Closed-form NASA-7 reference relations (in double) --------------------
double nasa7_cp(const Input<1>& in, double T)
{
    const double R = ge::ideal_gas_constant<double>;
    return R * (in.a0 + (in.a1 * T) + (in.a2 * T * T) + (in.a3 * T * T * T) + (in.a4 * T * T * T * T));
}

double nasa7_enthalpy(const Input<1>& in, double T)
{
    const double R = ge::ideal_gas_constant<double>;
    return R * T *
           ((in.a4 / 5 * T * T * T * T) + (in.a3 / 4 * T * T * T) + (in.a2 / 3 * T * T) + (in.a1 / 2 * T) + in.a0 +
            (in.a5 / T));
}

// Standard-state entropy: the model entropy at the reference concentration c_ref.
double nasa7_entropy_std(const Input<1>& in, double T)
{
    const double R = ge::ideal_gas_constant<double>;
    return R * ((in.a4 / 4 * T * T * T * T) + (in.a3 / 3 * T * T * T) + (in.a2 / 2 * T * T) + (in.a1 * T) +
                (in.a0 * std::log(T)) + in.a6);
}

} // namespace

int main()
{
    suite<"nasa7"> nasa7 = [] {
        const double R = ge::ideal_gas_constant<double>;

        // ===================================================================
        // Generic structural consistency (delegated to the shared helper):
        //   Psi == c*a   and   Psi == sum_i Psi_i.
        // ===================================================================
        "helmholtz consistency (generic)"_test = [&] {
            const ge::Nasa7<2> model(binary_inputs);
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
            auto eos = make_nasa7_eos<2>(binary_inputs);
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
            auto eos = make_nasa7_eos<2>(binary_inputs);
            for (const double c : {30.0, 90.0, 175.0}) {
                for (const double T : {285.0, 360.0}) {
                    check_euler_pressure<2>(eos, {0.4 * c, 0.6 * c}, T);
                }
            }
        };

        // ===================================================================
        // NASA7-specific caloric properties vs the closed-form NASA-7 relations
        // at the reference state (T = T_ref, c = c_ref). This is the guard that
        // the Helmholtz formula itself reproduces the intended polynomial.
        // ===================================================================
        "single-species reference state"_test = [&] {
            auto eos = make_nasa7_eos<1>(unary_inputs);
            const Input<1>& in = unary_inputs[0];
            const double c_ref = p_ref / (R * T_ref);
            const std::array<double, 1> x{1.0};
            const std::span<const double, 1> xs{x};

            check_rel("calc_cp       == NASA7 c_p", ge::calc_cp(eos, c_ref, xs, T_ref), nasa7_cp(in, T_ref), 1e-9);
            check_rel("calc_enthalpy == NASA7 h", ge::calc_enthalpy(eos, c_ref, xs, T_ref), nasa7_enthalpy(in, T_ref),
                      1e-9);
            check_rel("calc_entropy  == NASA7 s", ge::calc_entropy(eos, c_ref, xs, T_ref), nasa7_entropy_std(in, T_ref),
                      1e-9);
        };

        // ===================================================================
        // NASA7-specific caloric properties off the reference state: the
        // polynomial relations hold at any T. The standard-state entropy is
        // referenced to the standard pressure p_ref, so at a general state the
        // entropy carries the ideal-gas pressure term -R ln(p / p_ref), with
        // p = c R T  =>  p / p_ref = (c T) / (c_ref T_ref). (Enthalpy and c_p
        // are pressure/concentration independent for an ideal gas.)
        // ===================================================================
        "single-species off-reference caloric"_test = [&] {
            auto eos = make_nasa7_eos<1>(unary_inputs);
            const Input<1>& in = unary_inputs[0];
            const double c_ref = p_ref / (R * T_ref);
            const std::array<double, 1> x{1.0};
            const std::span<const double, 1> xs{x};

            for (const double T : {250.0, 350.0, 600.0}) {
                for (const double c : {c_ref, 0.5 * c_ref, 3.0 * c_ref}) {
                    const double h_expected = nasa7_enthalpy(in, T);
                    const double cp_expected = nasa7_cp(in, T);
                    const double s_expected = nasa7_entropy_std(in, T) - (R * std::log((c * T) / (c_ref * T_ref)));
                    check_rel("calc_enthalpy(T)", ge::calc_enthalpy(eos, c, xs, T), h_expected, 1e-9);
                    check_rel("calc_cp(T,c)", ge::calc_cp(eos, c, xs, T), cp_expected, 1e-9);
                    check_rel("calc_entropy(T,c)", ge::calc_entropy(eos, c, xs, T), s_expected, 1e-9);
                }
            }
        };

        // ===================================================================
        // Full derivative-consistency sweep: every property in
        // core_calculations.hpp checked against 4th-order finite differences of
        // the model's own Helmholtz energy (delegated to the shared harness).
        // Covers both single-component and mixture states.
        // ===================================================================
        "derivative consistency vs finite differences"_test = [&] {
            auto unary = make_nasa7_eos<1>(unary_inputs);
            run_derivative_consistency_tests<1>(unary, 120.0, {1.0}, 310.0);
            run_derivative_consistency_tests<1>(unary, 200.0, {1.0}, 360.0);

            auto binary = make_nasa7_eos<2>(binary_inputs);
            run_derivative_consistency_tests<2>(binary, 100.0, {0.4, 0.6}, 300.0);
            run_derivative_consistency_tests<2>(binary, 250.0, {0.7, 0.3}, 350.0);
            run_derivative_consistency_tests<2>(binary, 40.0, {0.5, 0.5}, 280.0);
        };
    };
}
