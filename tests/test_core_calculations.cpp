#include "derivative_test_harness.hpp"
#include "eos_test_models.hpp"
#include "eoslab/core/core_calculations.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cmath>
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
            glis::eos::calc_chemical_potential(binary, rhos, T, std::span<double, 2>{mu});
            check_rel("mu[0]", mu[0], mu_exact[0], 1e-11);
            check_rel("mu[1]", mu[1], mu_exact[1], 1e-11);
        };

        // -------------------------------------------------------------------
        // calc_helmholtz_density (Psi) consistency.
        //
        // The scalar Psi(rho, T) must agree with two independent references:
        //   (1) the sum of the per-component decomposition calc_partial_helmholtz
        //       (Psi = sum_i Psi_i), and
        //   (2) the molar Helmholtz energy via Psi = c * a(c, x, T), with
        //       c = sum_i rho_i and x_i = rho_i / c.
        // Checked for both the ideal and residual contributions, unary + binary.
        // -------------------------------------------------------------------
        "helmholtz_density consistency"_test = [&] {
            auto check_model = [&](const auto& model, auto rho) {
                const double T = 320.0;
                double c = 0.0;
                for (const double r : rho) {
                    c += r;
                }
                auto x = rho; // same type/size; converted to mole fractions
                for (auto& xi : x) {
                    xi /= c;
                }

                const double psi = model.calc_helmholtz_density(rho.data(), T);

                auto psi_i = rho; // reuse as scratch of the right size
                model.calc_partial_helmholtz(rho.data(), T, psi_i.data());
                double psi_sum = 0.0;
                for (const double v : psi_i) {
                    psi_sum += v;
                }
                check_rel("Psi == sum_i Psi_i", psi, psi_sum, 1e-12);

                const double a = model.calc_helmholtz(c, x.data(), T);
                check_rel("Psi == c * a", psi, c * a, 1e-12);
            };

            check_model(binary.ideal(), std::array<double, 2>{60.0, 90.0});
            check_model(binary.residual(), std::array<double, 2>{60.0, 90.0});
            check_model(unary.ideal(), std::array<double, 1>{120.0});
            check_model(unary.residual(), std::array<double, 1>{120.0});
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
