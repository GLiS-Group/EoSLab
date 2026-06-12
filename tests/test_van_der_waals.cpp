//
// Unit tests for the van der Waals residual model (glis::eos::VanDerWaals).
//
// Structure mirrors test_nasa7.cpp: the model-agnostic structural and
// derivative checks are delegated to derivative_test_harness.hpp, and what
// remains here is vdW-specific:
//   - the residual Helmholtz energy against an independent long-double
//     reference built directly from the spec formulas
//       a0_ii = 27 (R T_c)^2 / (64 P_c),   b_ii = R T_c / (8 P_c),
//       a_ij  = (1 - k_ij) sqrt(a0_ii a0_jj)   (vdW: T-independent, m_ii = 0),
//       a_m   = sum_ij x_i x_j a_ij,           b_m = sum_i x_i b_ii,
//       a_r   = -R T ln(1 - b_m c) - a_m c     (psi_2 = c for vdW)
//   - the critical-point identities p(T_c, c_c) = P_c and dp/dc = 0 at
//     c_c = 1 / (3 b), which pin down the a0/b parameter construction.
//
#include "derivative_test_harness.hpp"
#include "eoslab/core/core_calculations.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/ideal_models/const_cp.hpp"
#include "eoslab/residual_models/van_der_waals.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

using namespace boost::ut;
using namespace eoslab_test;

namespace {

namespace ge = glis::eos;

template<std::size_t N> using Input = typename ge::VanDerWaals<N>::SpeciesInput;

// N2 and CO2 critical data.
constexpr Input<2> n2{.T_c = 126.192, .P_c = 3.3958e6};
constexpr Input<2> co2{.T_c = 304.1282, .P_c = 7.3773e6};

constexpr std::array<Input<2>, 2> binary_inputs{{n2, co2}};
constexpr std::array<Input<1>, 1> unary_inputs{{{.T_c = n2.T_c, .P_c = n2.P_c}}};

// Asymmetric on purpose: only the symmetrized average can affect a_m, and the
// reference below applies each k_ij literally in the full double sum.
constexpr std::array<double, 4> binary_kij{0.0, 0.10, 0.04, 0.0};

// A ConstantCp ideal part so the residual can be paired into a full EoS.
template<std::size_t N> auto make_ideal()
{
    std::array<typename ge::ConstantCp<N>::SpeciesInput, N> in{};
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = {.T_ref = 298.15,
                 .p_ref = 1.0e5,
                 .c_p = 29.1 + (2.0 * static_cast<double>(i)),
                 .h_ref = 1000.0 * static_cast<double>(i),
                 .s_ref = 150.0 + (10.0 * static_cast<double>(i))};
    }
    return ge::ConstantCp<N>(in);
}

// --- Independent long-double reference, straight from the spec -------------
constexpr long double Rld = ge::ideal_gas_constant<long double>;

long double vdw_a0(const long double T_c, const long double P_c)
{
    return 27.0L * (Rld * T_c) * (Rld * T_c) / (64.0L * P_c);
}

long double vdw_b(const long double T_c, const long double P_c) { return Rld * T_c / (8.0L * P_c); }

// Molar residual Helmholtz energy a_r [J/mol] for n species.
template<std::size_t N>
long double ref_helmholtz(const std::array<Input<N>, N>& in, const std::array<double, N * N>& kij, long double c,
                          const std::array<double, N>& x, long double T)
{
    long double am = 0.0L;
    long double bm = 0.0L;
    for (std::size_t i = 0; i < N; ++i) {
        bm += static_cast<long double>(x[i]) * vdw_b(in[i].T_c, in[i].P_c);
        for (std::size_t j = 0; j < N; ++j) {
            const long double aij = (1.0L - static_cast<long double>(kij[(i * N) + j])) *
                                    std::sqrt(vdw_a0(in[i].T_c, in[i].P_c) * vdw_a0(in[j].T_c, in[j].P_c));
            am += static_cast<long double>(x[i]) * static_cast<long double>(x[j]) * aij;
        }
    }
    return (-Rld * T * std::log(1.0L - (bm * c))) - (am * c);
}

} // namespace

int main()
{
    suite<"van_der_waals"> vdw_suite = [] {
        // ===================================================================
        // Pure species: a_r against the closed-form reference over a sweep of
        // states. Pins down a0_ii, b_ii, and the psi_1/psi_2 assembly.
        // ===================================================================
        "pure species residual helmholtz matches closed form"_test = [] {
            const ge::VanDerWaals<1> model(unary_inputs);
            for (const double c : {1.0, 100.0, 5000.0, 15000.0}) {
                for (const double T : {120.0, 300.0, 500.0}) {
                    const long double ref = ref_helmholtz<1>(unary_inputs, {0.0}, c, {1.0}, T);
                    const std::array<double, 1> x{1.0};
                    check_rel("a_r (pure)", model.calc_helmholtz(c, x.data(), T), static_cast<double>(ref), 1e-12);
                }
            }
        };

        // ===================================================================
        // Binary mixture with asymmetric k_ij: the reference applies k_12 and
        // k_21 literally in the full double sum, so this verifies both the
        // mixing rules and that the implementation's internal symmetrization
        // is exact.
        // ===================================================================
        "binary mixture with asymmetric kij matches closed form"_test = [] {
            const ge::VanDerWaals<2> model(binary_inputs, binary_kij);
            for (const double c : {50.0, 2000.0, 9000.0}) {
                for (const double T : {250.0, 320.0, 450.0}) {
                    for (const std::array<double, 2> x : {std::array{0.3, 0.7}, std::array{0.85, 0.15}}) {
                        const long double ref = ref_helmholtz<2>(binary_inputs, binary_kij, c, x, T);
                        check_rel("a_r (binary)", model.calc_helmholtz(c, x.data(), T), static_cast<double>(ref),
                                  1e-12);
                    }
                }
            }
        };

        // ===================================================================
        // Cross-validation against NIST's teqp library (v0.23.1): golden
        // values of alphar = a_r / (R T) computed with teqp.vdWEOS(Tc, pc)
        // .get_Ar00(T, rho, z) for the same critical data as above. Both
        // libraries use the full CODATA gas constant.
        // ===================================================================
        "alphar matches teqp reference values"_test = [] {
            struct Ref {
                double T;      // [K]
                double c;      // [mol/m^3]
                double alphar; // a_r / (R T) [-]
            };
            const double R = ge::ideal_gas_constant<double>;

            const ge::VanDerWaals<1> pure(unary_inputs); // N2
            constexpr std::array<Ref, 4> pure_refs{{
                {.T = 500.0, .c = 1.0, .alphar = 5.7246696369883375e-06},
                {.T = 300.0, .c = 100.0, .alphar = -0.0016133301849820926},
                {.T = 300.0, .c = 5000.0, .alphar = -0.059582547773369049},
                {.T = 120.0, .c = 15000.0, .alphar = -1.1902211525514714},
            }};
            const std::array<double, 1> x1{1.0};
            for (const Ref& r : pure_refs) {
                check_rel("alphar vs teqp (pure N2)", pure.calc_helmholtz(r.c, x1.data(), r.T) / (R * r.T), r.alphar,
                          1e-13);
            }

            // teqp's vdWEOS mixes with sqrt(a_i a_j) and no k_ij.
            const ge::VanDerWaals<2> binary(binary_inputs); // N2 + CO2, kij = 0
            constexpr std::array<Ref, 3> binary_refs{{
                {.T = 300.0, .c = 100.0, .alphar = -0.0063338824515883491},
                {.T = 250.0, .c = 2000.0, .alphar = -0.16509357279216791},
                {.T = 450.0, .c = 9000.0, .alphar = -0.1648018426592458},
            }};
            const std::array<double, 2> x2{0.4, 0.6};
            for (const Ref& r : binary_refs) {
                check_rel("alphar vs teqp (binary)", binary.calc_helmholtz(r.c, x2.data(), r.T) / (R * r.T), r.alphar,
                          1e-13);
            }
        };

        // ===================================================================
        // Omitting kij must equal passing an all-zero matrix.
        // ===================================================================
        "kij defaults to zero"_test = [] {
            const ge::VanDerWaals<2> defaulted(binary_inputs);
            const ge::VanDerWaals<2> zeros(binary_inputs, std::array<double, 4>{});
            const std::array<double, 2> x{0.4, 0.6};
            for (const double c : {100.0, 4000.0}) {
                check_rel("a_r (kij default)", defaulted.calc_helmholtz(c, x.data(), 300.0),
                          zeros.calc_helmholtz(c, x.data(), 300.0), 1e-15);
            }
        };

        // ===================================================================
        // Static and dynamic instances built from the same data must agree.
        // ===================================================================
        "static and dynamic instances agree"_test = [] {
            const ge::VanDerWaals<2> stat(binary_inputs, binary_kij);
            using DynInput = ge::VanDerWaals<>::SpeciesInput;
            std::vector<DynInput> in_dyn;
            for (const auto& in : binary_inputs) {
                in_dyn.push_back({.T_c = in.T_c, .P_c = in.P_c});
            }
            const std::vector<double> kij_dyn(binary_kij.begin(), binary_kij.end());
            const ge::VanDerWaals<> dyn{std::span<const DynInput>{in_dyn}, std::span<const double>{kij_dyn}};
            expect(dyn.size() == 2_ul);
            const std::array<double, 2> x{0.25, 0.75};
            for (const double c : {75.0, 6000.0}) {
                for (const double T : {220.0, 380.0}) {
                    check_rel("static == dynamic", dyn.calc_helmholtz(c, x.data(), T),
                              stat.calc_helmholtz(c, x.data(), T), 1e-15);
                }
            }
        };

        // ===================================================================
        // Generic structural consistency: Psi == c*a and Psi == sum_i Psi_i
        // (delegated to the shared helper).
        // ===================================================================
        "helmholtz consistency (generic)"_test = [] {
            const ge::VanDerWaals<2> model(binary_inputs, binary_kij);
            for (const double T : {240.0, 310.0, 420.0}) {
                check_helmholtz_consistency<2>(model, {40.0, 70.0}, T);
                check_helmholtz_consistency<2>(model, {1500.0, 3500.0}, T);
                check_helmholtz_consistency<2>(model, {8000.0, 1000.0}, T);
            }
        };

        // ===================================================================
        // Pressure-explicit form of vdW for a pure species:
        //     p = c R T / (1 - b c) - a c^2
        // The framework obtains p from the Helmholtz residual via autodiff, so
        // agreement here verifies the a_r assembly against the textbook EoS.
        // ===================================================================
        "pure species pressure matches pressure-explicit form"_test = [] {
            const ge::EoS eos{make_ideal<1>(), ge::VanDerWaals<1>(unary_inputs)};
            const double R = ge::ideal_gas_constant<double>;
            const double a = static_cast<double>(vdw_a0(n2.T_c, n2.P_c));
            const double b = static_cast<double>(vdw_b(n2.T_c, n2.P_c));
            const std::array<double, 1> x{1.0};
            const std::span<const double, 1> xs{x};
            for (const double c : {1.0, 100.0, 5000.0, 15000.0}) {
                for (const double T : {120.0, 300.0, 500.0}) {
                    const double p_ref = (c * R * T / (1.0 - (b * c))) - (a * c * c);
                    check_rel("p == cRT/(1-bc) - a c^2", ge::calc_pressure(eos, c, xs, T), p_ref, 1e-9);
                }
            }
        };

        // ===================================================================
        // vdW critical-point identities for a pure species: at T = T_c and
        // c_c = 1/(3 b) the model must reproduce p = P_c with dp/dc = 0.
        // This is the strongest check that a0 and b were formed correctly.
        // ===================================================================
        "pure species reproduces its critical point"_test = [] {
            const ge::EoS eos{make_ideal<1>(), ge::VanDerWaals<1>(unary_inputs)};
            const double b = static_cast<double>(vdw_b(n2.T_c, n2.P_c));
            const double c_c = 1.0 / (3.0 * b);
            const std::array<double, 1> x{1.0};
            const std::span<const double, 1> xs{x};
            check_rel("p(T_c, c_c) == P_c", ge::calc_pressure(eos, c_c, xs, n2.T_c), n2.P_c, 1e-9);
            const double dpdc = ge::calc_dp_dc(eos, c_c, xs, n2.T_c);
            // dp/dc vanishes at the critical point; scale by R*T_c since the
            // individual terms it cancels from are O(R*T_c).
            expect(std::abs(dpdc) <= 1e-6 * ge::ideal_gas_constant<double> * n2.T_c)
                << "dp/dc at critical point: " << dpdc;
        };

        // ===================================================================
        // Pressure via the Euler relation (mixture, reverse-mode chemical
        // potentials) and the full derivative-consistency harness.
        // ===================================================================
        "pressure via Euler relation (mixture)"_test = [] {
            const ge::EoS eos{make_ideal<2>(), ge::VanDerWaals<2>(binary_inputs, binary_kij)};
            for (const double c : {30.0, 900.0, 7000.0}) {
                for (const double T : {260.0, 360.0}) {
                    check_euler_pressure<2>(eos, {0.4 * c, 0.6 * c}, T);
                }
            }
        };

        "derivative consistency (pure, gas and dense)"_test = [] {
            const ge::EoS eos{make_ideal<1>(), ge::VanDerWaals<1>(unary_inputs)};
            run_derivative_consistency_tests<1>(eos, 100.0, {1.0}, 300.0, 0.028);
            run_derivative_consistency_tests<1>(eos, 8000.0, {1.0}, 320.0, 0.028);
        };

        "derivative consistency (binary mixture)"_test = [] {
            const ge::EoS eos{make_ideal<2>(), ge::VanDerWaals<2>(binary_inputs, binary_kij)};
            run_derivative_consistency_tests<2>(eos, 150.0, {0.3, 0.7}, 310.0, 0.036);
            run_derivative_consistency_tests<2>(eos, 5000.0, {0.6, 0.4}, 350.0, 0.033);
        };
    };
}
