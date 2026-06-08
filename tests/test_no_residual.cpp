//
// Tests for glis::eos::NoResidual: a residual contribution that is identically
// zero. Pairing it with an ideal-gas model yields a pure ideal gas.
//
// Two size regimes are exercised:
//   * compile-time size (N is a concrete value), and
//   * runtime size       (N == std::dynamic_extent, count passed to the ctor).
//
// The compile-time case is additionally run through the full Enzyme-backed
// derivative-consistency harness (paired with IdealGasTestModel). That doubles
// as a regression guard: if Enzyme were to mis-handle the (storage-free)
// NoResidual class, the autodiff'd residual derivatives would diverge from the
// analytic ideal-gas references.
//
#include "derivative_test_harness.hpp"
#include "eos_test_models.hpp"
#include "eoslab/core/core_calculations.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/residual_models/no_residual.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <span>
#include <tuple>

using namespace boost::ut;
using namespace eoslab_test;
using glis::eos::NoResidual;

int main()
{
    suite<"no_residual"> s = [] {
        // -------------------------------------------------------------------
        // Compile-time known size: default-constructible, fixed size().
        // -------------------------------------------------------------------
        "compile-time size: construction and zero output"_test = []<typename Number> {
            constexpr std::size_t N = 3;
            const NoResidual<N> res{};            // only a default ctor is needed
            expect(eq(NoResidual<N>::size(), N)); // size() is static for a compile-time size

            const std::array<Number, N> rho{Number{1.0}, Number{2.0}, Number{3.0}};
            const Number c{6.0};
            const Number T{300.0};

            expect(eq(res.calc_helmholtz(c, rho.data(), T), Number{0}));
            expect(eq(res.calc_helmholtz_density(rho.data(), T), Number{0}));

            // Pre-fill with garbage to confirm every entry is overwritten.
            std::array<Number, N> out{Number{7}, Number{-8}, Number{9}};
            res.calc_partial_helmholtz(rho.data(), T, out.data());
            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(out[i], Number{0}));
            }
        } | std::tuple<float, double, long double>{};

        // -------------------------------------------------------------------
        // Runtime known size: size passed to the constructor, then honoured by
        // size() and by calc_partial_helmholtz's loop bound.
        // -------------------------------------------------------------------
        "runtime size: construction and zero output"_test = []<typename Number> {
            constexpr std::size_t N = 4;
            const NoResidual<std::dynamic_extent> res{N}; // runtime size to ctor
            expect(eq(res.size(), N));

            const std::array<Number, N> rho{Number{1.0}, Number{2.0}, Number{3.0}, Number{4.0}};
            const Number c{10.0};
            const Number T{275.0};

            expect(eq(res.calc_helmholtz(c, rho.data(), T), Number{0}));
            expect(eq(res.calc_helmholtz_density(rho.data(), T), Number{0}));

            std::array<Number, N> out{Number{7}, Number{-8}, Number{9}, Number{-1}};
            res.calc_partial_helmholtz(rho.data(), T, out.data());
            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(out[i], Number{0}));
            }
        } | std::tuple<float, double, long double>{};

        // -------------------------------------------------------------------
        // Paired with an ideal-gas model the residual vanishes, so the EoS is a
        // pure ideal gas: Z = p / (c R T) == 1 *exactly* (no dilute limit
        // needed, unlike the virial model). This routes Enzyme through the
        // storage-free NoResidual class.
        // -------------------------------------------------------------------
        "ideal gas pairing: Z == 1"_test = [] {
            const double R = glis::eos::ideal_gas_constant<double>;
            IdealGasTestModel<2> ideal{{2.5, 3.1}, {1.5, 2.0}};
            NoResidual<2> residual{};
            glis::eos::EoS<IdealGasTestModel<2>, NoResidual<2>> eos{ideal, residual};

            const std::array<double, 2> x{0.4, 0.6};
            std::span<const double, 2> xs{x};
            for (const double c : {1.0, 100.0, 5000.0}) {
                for (const double T : {200.0, 350.0, 500.0}) {
                    const double Z = glis::eos::calc_pressure(eos, c, xs, T) / (c * R * T);
                    expect(std::abs(Z - 1.0) < 1e-12) << "Z =" << Z;
                }
            }
        };

        // -------------------------------------------------------------------
        // Full derivative-consistency harness on the ideal-gas pairing. With a
        // zero residual all residual derivatives are zero and the totals reduce
        // to the analytic ideal-gas references; a passing run confirms Enzyme
        // differentiates the empty residual correctly.
        // -------------------------------------------------------------------
        "ideal gas pairing: derivative consistency"_test = [] {
            IdealGasTestModel<2> ideal{{2.5, 3.1}, {1.5, 2.0}};
            NoResidual<2> residual{};
            glis::eos::EoS<IdealGasTestModel<2>, NoResidual<2>> eos{ideal, residual};

            run_derivative_consistency_tests<2>(eos, 100.0, {0.4, 0.6}, 300.0);
            run_derivative_consistency_tests<2>(eos, 250.0, {0.7, 0.3}, 350.0);
        };
    };
}
