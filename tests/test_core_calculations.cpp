#include "derivative_test_harness.hpp"
#include "eos_test_models.hpp"
#include "eoslab/core_calculations.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <limits>
#include <span>

using namespace boost::ut;
using namespace eoslab_test;

namespace {

// Independent, fully analytic references for the virial residual + ideal-gas
// model pair. These do not use finite differences, so they catch sign / formula
// errors that an autodiff-vs-FD comparison alone might miss.
struct BinaryAnalytic {
    static constexpr std::array<double, 2> b{1.0e-3, 1.5e-3};
    static constexpr std::array<double, 2> beta{1.0e-1, 8.0e-2};
    static constexpr std::array<double, 2> gamma{2.0e-6, 1.0e-6};
    static constexpr std::array<double, 2> h{2.5, 3.1};
    static constexpr std::array<double, 2> g{1.5, 2.0};

    static double Bmix(std::array<double, 2> x, double T)
    {
        return (x[0] * (b[0] - (beta[0] / T))) + (x[1] * (b[1] - (beta[1] / T)));
    }
    static double Cmix(std::array<double, 2> x) { return (x[0] * gamma[0]) + (x[1] * gamma[1]); }
};

} // namespace

int main()
{
    // NOTE: boost::ut requires the suite lambda to be captureless (it is
    // converted to a function pointer), so shared state lives inside it.
    suite<"core_calculations"> core = [] {
        const double R = glis::eos::ideal_gas_constant<double>;
        auto binary = make_binary_model();
        auto unary = make_unary_model();

        // -------------------------------------------------------------------
        // Generic finite-difference derivative-consistency harness.
        // -------------------------------------------------------------------
        "binary derivative consistency"_test = [&] {
            run_derivative_consistency_tests<2>(binary, 100.0, {0.4, 0.6}, 300.0);
            run_derivative_consistency_tests<2>(binary, 250.0, {0.7, 0.3}, 350.0);
            run_derivative_consistency_tests<2>(binary, 50.0, {0.5, 0.5}, 280.0);
            run_derivative_consistency_tests<2>(binary, 300.0, {0.2, 0.8}, 400.0);
        };

        "unary derivative consistency"_test = [&] {
            run_derivative_consistency_tests<1>(unary, 120.0, {1.0}, 310.0);
            run_derivative_consistency_tests<1>(unary, 200.0, {1.0}, 360.0);
        };

        // -------------------------------------------------------------------
        // Independent closed-form checks (virial EoS).
        // -------------------------------------------------------------------
        "virial pressure analytic"_test = [&] {
            const std::array<double, 2> x{0.35, 0.65};
            for (double c : {40.0, 120.0, 260.0}) {
                for (double T : {270.0, 330.0, 410.0}) {
                    const double B = BinaryAnalytic::Bmix(x, T);
                    const double C = BinaryAnalytic::Cmix(x);
                    const double p_exact = R * T * (c + (B * c * c) + (C * c * c * c));
                    const double dpdc_exact = R * T * (1.0 + (2.0 * B * c) + (3.0 * C * c * c));
                    std::span<const double, 2> xs{x};
                    check_rel("p_virial", glis::eos::calc_pressure(binary, c, xs, T), p_exact, 1e-12);
                    check_rel("dp_dc_virial", glis::eos::calc_dp_dc(binary, c, xs, T), dpdc_exact, 1e-12);
                }
            }
        };

        "chemical potential analytic"_test = [&] {
            const std::array<double, 2> x{0.45, 0.55};
            const double c = 150.0;
            const double T = 320.0;
            std::array<double, 2> rho{x[0] * c, x[1] * c};
            // analytic mu_i = mu_i^ideal + mu_i^res
            const double S1 = (rho[0] * (BinaryAnalytic::b[0] - (BinaryAnalytic::beta[0] / T))) +
                              (rho[1] * (BinaryAnalytic::b[1] - (BinaryAnalytic::beta[1] / T)));
            const double S2 = (rho[0] * BinaryAnalytic::gamma[0]) + (rho[1] * BinaryAnalytic::gamma[1]);
            std::array<double, 2> mu_exact{};
            for (std::size_t i = 0; i < 2; ++i) {
                const double phi = BinaryAnalytic::h[i] - (BinaryAnalytic::g[i] * std::log(T));
                const double mu_ideal = R * T * (std::log(rho[i]) + phi + 1.0);
                const double mu_res = R * T *
                                      (S1 + (c * (BinaryAnalytic::b[i] - (BinaryAnalytic::beta[i] / T))) + (c * S2) +
                                       (0.5 * c * c * BinaryAnalytic::gamma[i]));
                mu_exact[i] = mu_ideal + mu_res;
            }
            std::span<const double, 2> rhos{rho};
            std::array<double, 2> mu{};
            std::array<double, 2> scratch1{};
            std::array<double, 2> scratch2{};
            glis::eos::calc_chemical_potential(binary, rhos, T, std::span<double, 2>{mu},
                                               std::span<double, 2>{scratch1}, std::span<double, 2>{scratch2});
            check_rel("mu[0]", mu[0], mu_exact[0], 1e-11);
            check_rel("mu[1]", mu[1], mu_exact[1], 1e-11);
        };

        // -------------------------------------------------------------------
        // Scratch buffers must NOT require caller pre-initialisation.
        //
        // The partial-molar routines take two caller-allocated scratch buffers
        // (scratch1 = primal Psi_i, scratch2 = Enzyme shadow). This test pokes
        // garbage -- including NaN/Inf -- into both buffers before each call and
        // checks the results are unchanged. The internal handling is: scratch1
        // is fully overwritten before use, and scratch2 is zeroed internally
        // (the Enzyme reverse pass requires a zeroed shadow). A passing run
        // confirms the @note in the docs: the user need not preprocess either
        // array.
        //
        // NOTE on tolerance: we compare against a clean reference with a *tight*
        // relative tolerance rather than exact equality. The clean reference and
        // the dirty calls are distinct call sites; with the reference's scratch
        // known to be zero at compile time the optimizer/Enzyme may contract
        // floating-point ops slightly differently, giving a ~1 ULP (~1e-16
        // relative) difference. That is codegen noise, not a dependence on the
        // scratch contents -- a real dependence (e.g. an unzeroed shadow) instead
        // corrupts the result by many orders of magnitude or yields NaN.
        // -------------------------------------------------------------------
        "scratch buffers need no initialisation"_test = [&] {
            const std::array<double, 2> x{0.45, 0.55};
            const double c = 150.0;
            const double T = 320.0;
            const std::array<double, 2> rho{x[0] * c, x[1] * c};
            std::span<const double, 2> xs{x};
            std::span<const double, 2> rhos{rho};

            const double inf = std::numeric_limits<double>::infinity();
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const std::array<std::array<double, 2>, 3> garbage{{{1.0e300, -7.0e-12}, {inf, -inf}, {nan, nan}}};

            // Reference values from a clean (zero-initialised) run.
            std::array<double, 2> mu_ref{};
            std::array<double, 2> logphi_ref{};
            std::array<double, 2> fug_ref{};
            {
                std::array<double, 2> s1{};
                std::array<double, 2> s2{};
                glis::eos::calc_chemical_potential(binary, rhos, T, std::span<double, 2>{mu_ref},
                                                   std::span<double, 2>{s1}, std::span<double, 2>{s2});
                glis::eos::calc_log_fugacity_coeff(binary, c, xs, T, rhos, std::span<double, 2>{logphi_ref},
                                                   std::span<double, 2>{s1}, std::span<double, 2>{s2});
                glis::eos::calc_fugacity(binary, rhos, T, std::span<double, 2>{fug_ref}, std::span<double, 2>{s1},
                                         std::span<double, 2>{s2});
            }

            constexpr double tol = 1e-12; // far above ~1 ULP, far below any real corruption
            for (const auto& g : garbage) {
                auto s1 = g; // dirty scratch1
                auto s2 = g; // dirty scratch2
                std::array<double, 2> mu{};
                glis::eos::calc_chemical_potential(binary, rhos, T, std::span<double, 2>{mu}, std::span<double, 2>{s1},
                                                   std::span<double, 2>{s2});
                check_rel("mu[0] (dirty scratch)", mu[0], mu_ref[0], tol);
                check_rel("mu[1] (dirty scratch)", mu[1], mu_ref[1], tol);

                s1 = g;
                s2 = g;
                std::array<double, 2> logphi{};
                glis::eos::calc_log_fugacity_coeff(binary, c, xs, T, rhos, std::span<double, 2>{logphi},
                                                   std::span<double, 2>{s1}, std::span<double, 2>{s2});
                check_rel("logphi[0] (dirty scratch)", logphi[0], logphi_ref[0], tol);
                check_rel("logphi[1] (dirty scratch)", logphi[1], logphi_ref[1], tol);

                s1 = g;
                s2 = g;
                std::array<double, 2> fug{};
                glis::eos::calc_fugacity(binary, rhos, T, std::span<double, 2>{fug}, std::span<double, 2>{s1},
                                         std::span<double, 2>{s2});
                check_rel("fug[0] (dirty scratch)", fug[0], fug_ref[0], tol);
                check_rel("fug[1] (dirty scratch)", fug[1], fug_ref[1], tol);
            }
        };

        // -------------------------------------------------------------------
        // Sanity: ideal-gas limit (residual contributions vanish as c -> 0).
        // At low density the compressibility factor Z = p / (c R T) -> 1.
        // -------------------------------------------------------------------
        "ideal gas limit"_test = [&] {
            const std::array<double, 2> x{0.5, 0.5};
            const double T = 300.0;
            const double c = 1.0e-3; // very dilute
            std::span<const double, 2> xs{x};
            const double Z = glis::eos::calc_pressure(binary, c, xs, T) / (c * R * T);
            expect(std::abs(Z - 1.0) < 1e-5) << "Z should approach 1 in the dilute limit, got" << Z;
        };
    };
}
