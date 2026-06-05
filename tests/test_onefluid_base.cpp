//
// Tests for glis::eos::OneFluidBase: the CRTP orchestrator for one-fluid models,
// where a mixing rule (pre-calculation) averages the per-species parameters and
// a single bulk kernel evaluates the pseudo-pure fluid.
//
// Test model (a residual contribution): linear mixing b_bar(x) = sum_i x_i b_i,
// with bulk reduced residual alpha^r = b_bar c, i.e.
//   a_res(c, x, T)   = R T b_bar c
//   Psi_res(rho, T)  = c a_res = R T c^2 b_bar      (c = sum rho_i)
// Hand-derivable, so it checks the closed form, the mixing rule, Psi = c*a, the
// SoA column-averaging path at large N, and Enzyme through the bulk expression.
//
#include "derivative_test_harness.hpp"
#include "eos_test_models.hpp"
#include "eoslab/core/concepts.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/core/onefluid_base.hpp"

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

using namespace boost::ut;
namespace ge = glis::eos;

// Named (not anonymous) namespace: Enzyme requires the differentiated model type
// to have linkage.
namespace of_test {

void check_close(std::string_view name, double actual, double expected, double rtol = 1e-12)
{
    const double denom = std::max(1.0, std::abs(expected));
    expect(std::abs(actual - expected) / denom <= rtol) << name << ": actual=" << actual << " expected=" << expected;
}

struct OfParams {
    double b;
};

template<class Number> struct Avg {
    Number b_bar;
};

template<std::size_t N> class OfLinearResidual : public ge::OneFluidBase<OfLinearResidual<N>, OfParams, N> {
public:
    using Base = ge::OneFluidBase<OfLinearResidual<N>, OfParams, N>;
    using Base::Base;

    template<std::floating_point Number>
    [[nodiscard]] Avg<Number> perform_pre_calculations(Number /*c*/, const Number* x, Number /*T*/) const
    {
        const auto b_col = this->column(0); // contiguous SoA column (averaging path)
        Number b_bar{0};
        this->for_each_component([&](std::size_t i) { b_bar += x[i] * Number(b_col[i]); });
        return Avg<Number>{b_bar};
    }

    template<std::floating_point Number>
    [[nodiscard]] Avg<Number> perform_pre_calculations(const Number* rho_i, Number /*T*/) const
    {
        const auto b_col = this->column(0);
        Number c{0};
        Number sum_rb{0};
        this->for_each_component([&](std::size_t i) {
            c += rho_i[i];
            sum_rb += rho_i[i] * Number(b_col[i]);
        });
        return Avg<Number>{sum_rb / c};
    }

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_bulk(Number c, const Number* /*x*/, Number T, const Avg<Number>& avg) const
    {
        const Number R = ge::ideal_gas_constant<Number>;
        return R * T * avg.b_bar * c;
    }

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density_bulk(const Number* rho_i, Number T, const Avg<Number>& avg) const
    {
        const Number R = ge::ideal_gas_constant<Number>;
        Number c{0};
        this->for_each_component([&](std::size_t i) { c += rho_i[i]; });
        return R * T * c * c * avg.b_bar;
    }
};

// Independent reference: a_res = R T (sum_i x_i b_i) c.
double res_a(double c, std::span<const double> x, double T, std::span<const OfParams> prm)
{
    const double R = ge::ideal_gas_constant<double>;
    double b_bar = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        b_bar += x[i] * prm[i].b;
    }
    return R * T * b_bar * c;
}

constexpr std::array<OfParams, 2> kParams{OfParams{1.0e-3}, OfParams{1.5e-3}};

} // namespace of_test

using namespace of_test;

static_assert(ge::EquationOfState<OfLinearResidual<2>>);
static_assert(ge::ResidualEoS<OfLinearResidual<2>>); // not an ideal contribution

int main()
{
    suite<"onefluid_base"> s = [] {
        const std::array<double, 2> x{0.35, 0.65};
        const std::span<const double> xs{x};
        const std::span<const OfParams> prm{kParams};

        // ----------------------------------------------------------------- //
        // Bulk calc_helmholtz matches the closed form (validates the mixing
        // rule b_bar = sum_i x_i b_i through the base).
        // ----------------------------------------------------------------- //
        "bulk calc_helmholtz matches closed form"_test = [&] {
            const OfLinearResidual<2> model{kParams};
            for (const double c : {40.0, 120.0, 300.0}) {
                for (const double T : {280.0, 350.0}) {
                    check_close("a_res", model.calc_helmholtz(c, x.data(), T), res_a(c, xs, T, prm));
                }
            }
        };

        // ----------------------------------------------------------------- //
        // Psi = c * a consistency.
        // ----------------------------------------------------------------- //
        "helmholtz_density equals c*a"_test = [&] {
            const OfLinearResidual<2> model{kParams};
            const double c = 120.0;
            const double T = 330.0;
            const std::array<double, 2> rho{x[0] * c, x[1] * c};
            check_close("Psi == c*a", model.calc_helmholtz_density(rho.data(), T),
                        c * model.calc_helmholtz(c, x.data(), T));
        };

        // ----------------------------------------------------------------- //
        // calc_partial_helmholtz entries sum to Psi (mole-fraction split).
        // ----------------------------------------------------------------- //
        "partial helmholtz sums to Psi"_test = [&] {
            const OfLinearResidual<2> model{kParams};
            const double c = 150.0;
            const double T = 300.0;
            const std::array<double, 2> rho{x[0] * c, x[1] * c};
            std::array<double, 2> out{};
            model.calc_partial_helmholtz(rho.data(), T, out.data());
            check_close("sum == Psi", out[0] + out[1], model.calc_helmholtz_density(rho.data(), T));
        };

        // ----------------------------------------------------------------- //
        // Dynamic-N agrees with fixed-N.
        // ----------------------------------------------------------------- //
        "dynamic-N matches fixed-N"_test = [&] {
            const OfLinearResidual<2> fixed{kParams};
            const std::vector<OfParams> rows{kParams[0], kParams[1]};
            const OfLinearResidual<std::dynamic_extent> dyn{std::span<const OfParams>{rows}};
            const double c = 175.0;
            const double T = 305.0;
            expect(eq(dyn.size(), std::size_t{2}));
            check_close("a_res", dyn.calc_helmholtz(c, x.data(), T), fixed.calc_helmholtz(c, x.data(), T));
        };

        // ----------------------------------------------------------------- //
        // Large N: column-based averaging over many species.
        // ----------------------------------------------------------------- //
        "large-N column averaging"_test = [] {
            constexpr std::size_t M = 64;
            std::array<OfParams, M> rows{};
            std::array<double, M> x{};
            double bsum = 0.0;
            for (std::size_t i = 0; i < M; ++i) {
                rows[i] = OfParams{1.0e-3 + (1.0e-5 * static_cast<double>(i))};
                x[i] = 1.0 / static_cast<double>(M);
                bsum += x[i] * rows[i].b;
            }
            const OfLinearResidual<M> model{rows};
            const double c = 90.0;
            const double T = 300.0;
            const double R = ge::ideal_gas_constant<double>;
            check_close("a_res large N", model.calc_helmholtz(c, x.data(), T), R * T * bsum * c);
        };

        // ----------------------------------------------------------------- //
        // Enzyme forward + reverse mode through the single bulk expression.
        // Paired with the ideal-gas test model so the derivative-based
        // properties have a non-trivial reference.
        // ----------------------------------------------------------------- //
        "enzyme derivative consistency through the bulk kernel"_test = [] {
            eoslab_test::IdealGasTestModel<2> ideal{{2.5, 3.1}, {1.5, 2.0}};
            OfLinearResidual<2> residual{kParams};
            ge::EoS<eoslab_test::IdealGasTestModel<2>, OfLinearResidual<2>> eos{ideal, residual};

            eoslab_test::run_derivative_consistency_tests<2>(eos, 100.0, {0.4, 0.6}, 300.0);
            eoslab_test::run_derivative_consistency_tests<2>(eos, 250.0, {0.7, 0.3}, 350.0);
        };
    };
}
