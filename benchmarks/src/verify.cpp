//
// Correctness cross-checks for the refactored models and the new code paths the
// benchmarks exercise:
//   * SoA storage variants agree with their AoS originals on every property;
//   * the forward-mode gradient routines (`*_fwd`) agree with the reverse-mode
//     originals.
// Returns non-zero on the first mismatch so it can gate the benchmark build.
//
#include "eoslab/core/core_calculations.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/ideal_models/const_cp.hpp"
#include "eoslab/ideal_models/const_cp_soa.hpp"
#include "eoslab/ideal_models/nasa7.hpp"
#include "eoslab/ideal_models/nasa7_soa.hpp"
#include "eoslab/residual_models/no_residual.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>

namespace ge = glis::eos;

namespace {

int g_failures = 0;

void check(const std::string& what, double a, double b, double rtol = 1e-10)
{
    const double denom = std::max(1.0, std::abs(b));
    const double rel = std::abs(a - b) / denom;
    if (rel > rtol || std::isnan(rel)) {
        std::printf("  FAIL  %-44s  got=%.12g  ref=%.12g  rel=%.3g\n", what.c_str(), a, b, rel);
        ++g_failures;
    }
}

// Inputs (compile-time size 3).
constexpr std::size_t N = 3;

// Templated on the model so the (distinct) AoS and SoA nested SpeciesInput types
// are each filled from the same literal data.
template<class Model> std::array<typename Model::SpeciesInput, N> cc_inputs()
{
    using SI = typename Model::SpeciesInput;
    return {SI{.T_ref = 298.15, .p_ref = 1.0e5, .c_p = 29.1, .h_ref = -100.0, .s_ref = 191.0},
            SI{.T_ref = 300.0, .p_ref = 1.1e5, .c_p = 33.6, .h_ref = 50.0, .s_ref = 205.0},
            SI{.T_ref = 298.15, .p_ref = 0.9e5, .c_p = 37.1, .h_ref = -390.0, .s_ref = 213.0}};
}

template<class Model> std::array<typename Model::SpeciesInput, N> n7_inputs()
{
    using SI = typename Model::SpeciesInput;
    return {SI{.a0 = 3.53,
               .a1 = -1.2e-4,
               .a2 = -5.0e-7,
               .a3 = 2.4e-9,
               .a4 = -1.4e-12,
               .a5 = -1046.0,
               .a6 = 2.97,
               .T_ref = 298.15,
               .p_ref = 1.0e5},
            SI{.a0 = 3.78,
               .a1 = -3.0e-3,
               .a2 = 9.8e-6,
               .a3 = -9.7e-9,
               .a4 = 3.2e-12,
               .a5 = -1063.0,
               .a6 = 3.66,
               .T_ref = 300.0,
               .p_ref = 1.0e5},
            SI{.a0 = 2.50,
               .a1 = 7.0e-4,
               .a2 = -1.0e-6,
               .a3 = 8.0e-10,
               .a4 = -2.0e-13,
               .a5 = -900.0,
               .a6 = 4.10,
               .T_ref = 298.15,
               .p_ref = 1.0e5}};
}

const std::array<double, N> kx{0.5, 0.3, 0.2};
constexpr double kc = 120.0;
constexpr double kT = 350.0;
constexpr double kM = 0.029;

// Compare every scalar property + (rev vs fwd) gradients between two EoS pairs
// that should be numerically identical (AoS-vs-SoA), and rev-vs-fwd within one.
template<class EoSA, class EoSB> void compare(const std::string& tag, const EoSA& a, const EoSB& b)
{
    std::array<double, N> rho{};
    for (std::size_t i = 0; i < N; ++i) {
        rho[i] = kx[i] * kc;
    }
    const std::span<const double, N> xs{kx};
    const std::span<const double, N> rs{rho};

    check(tag + " helmholtz", ge::calc_helmholtz(a, kc, xs, kT), ge::calc_helmholtz(b, kc, xs, kT));
    check(tag + " pressure", ge::calc_pressure(a, kc, xs, kT), ge::calc_pressure(b, kc, xs, kT));
    check(tag + " internal_energy", ge::calc_internal_energy(a, kc, xs, kT), ge::calc_internal_energy(b, kc, xs, kT));
    check(tag + " enthalpy", ge::calc_enthalpy(a, kc, xs, kT), ge::calc_enthalpy(b, kc, xs, kT));
    check(tag + " entropy", ge::calc_entropy(a, kc, xs, kT), ge::calc_entropy(b, kc, xs, kT));
    check(tag + " gibbs", ge::calc_gibbs(a, kc, xs, kT), ge::calc_gibbs(b, kc, xs, kT));
    check(tag + " dp_dc", ge::calc_dp_dc(a, kc, xs, kT), ge::calc_dp_dc(b, kc, xs, kT));
    check(tag + " dp_dT", ge::calc_dp_dT(a, kc, xs, kT), ge::calc_dp_dT(b, kc, xs, kT));
    check(tag + " cv", ge::calc_cv(a, kc, xs, kT), ge::calc_cv(b, kc, xs, kT));
    check(tag + " cp", ge::calc_cp(a, kc, xs, kT), ge::calc_cp(b, kc, xs, kT));
    check(tag + " sound_speed_sq", ge::calc_sound_speed_squared(a, kc, xs, kT, kM),
          ge::calc_sound_speed_squared(b, kc, xs, kT, kM));

    std::array<double, N> mu_a{};
    std::array<double, N> mu_b{};
    ge::calc_chemical_potential(a, rs, kT, std::span<double, N>{mu_a});
    ge::calc_chemical_potential(b, rs, kT, std::span<double, N>{mu_b});
    for (std::size_t i = 0; i < N; ++i) {
        check(tag + " mu[" + std::to_string(i) + "]", mu_a[i], mu_b[i]);
    }
}

// rev-vs-fwd within a single EoS pair.
template<class EoSt> void compare_rev_fwd(const std::string& tag, const EoSt& e)
{
    std::array<double, N> rho{};
    for (std::size_t i = 0; i < N; ++i) {
        rho[i] = kx[i] * kc;
    }
    const std::span<const double, N> xs{kx};
    const std::span<const double, N> rs{rho};

    std::array<double, N> rev{};
    std::array<double, N> fwd{};

    ge::calc_chemical_potential(e, rs, kT, std::span<double, N>{rev});
    ge::calc_chemical_potential_fwd(e, rs, kT, std::span<double, N>{fwd});
    for (std::size_t i = 0; i < N; ++i) {
        check(tag + " mu rev/fwd[" + std::to_string(i) + "]", fwd[i], rev[i]);
    }

    ge::calc_fugacity(e, rs, kT, std::span<double, N>{rev});
    ge::calc_fugacity_fwd(e, rs, kT, std::span<double, N>{fwd});
    for (std::size_t i = 0; i < N; ++i) {
        check(tag + " fugacity rev/fwd[" + std::to_string(i) + "]", fwd[i], rev[i]);
    }

    ge::calc_log_fugacity_coeff(e, kc, xs, kT, rs, std::span<double, N>{rev});
    ge::calc_log_fugacity_coeff_fwd(e, kc, xs, kT, rs, std::span<double, N>{fwd});
    for (std::size_t i = 0; i < N; ++i) {
        check(tag + " lnphi rev/fwd[" + std::to_string(i) + "]", fwd[i], rev[i]);
    }
}

} // namespace

int main()
{
    using NoRes = ge::NoResidual<N>;

    const ge::ConstantCp<N> cc_aos(cc_inputs<ge::ConstantCp<N>>());
    const ge::ConstantCpSoA<N> cc_soa(cc_inputs<ge::ConstantCpSoA<N>>());
    const ge::Nasa7<N> n7_aos(n7_inputs<ge::Nasa7<N>>());
    const ge::Nasa7SoA<N> n7_soa(n7_inputs<ge::Nasa7SoA<N>>());

    const ge::EoS<ge::ConstantCp<N>, NoRes> eos_cc_aos{cc_aos, NoRes{}};
    const ge::EoS<ge::ConstantCpSoA<N>, NoRes> eos_cc_soa{cc_soa, NoRes{}};
    const ge::EoS<ge::Nasa7<N>, NoRes> eos_n7_aos{n7_aos, NoRes{}};
    const ge::EoS<ge::Nasa7SoA<N>, NoRes> eos_n7_soa{n7_soa, NoRes{}};

    std::puts("AoS vs SoA:");
    compare("ConstantCp AoS/SoA", eos_cc_aos, eos_cc_soa);
    compare("Nasa7      AoS/SoA", eos_n7_aos, eos_n7_soa);

    std::puts("reverse vs forward mode:");
    compare_rev_fwd("ConstantCp", eos_cc_aos);
    compare_rev_fwd("Nasa7", eos_n7_aos);

    if (g_failures == 0) {
        std::puts("ALL CHECKS PASSED");
        return EXIT_SUCCESS;
    }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return EXIT_FAILURE;
}
