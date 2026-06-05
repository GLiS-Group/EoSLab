//
// Tests for glis::eos::MultiFluidBase: the CRTP orchestrator that turns a pair
// of per-species kernels (+ optional pre-calculation) into a full EoS model by
// owning the component loop and pre-calc hoisting.
//
// Two test models, both an ideal gas a = R T [ln c + sum_i x_i (ln x_i + phi_i)]
// with phi_i = h_i - g_i ln T:
//   * MfIdealGas    -- no pre-calculation (kernels omit the pre argument),
//   * MfIdealGasPre -- caches ln T in a Number-templated pre-calc struct and
//                      counts how often the pre-calc runs.
//
// The Enzyme derivative-harness validation lives in test_no_residual-style
// coverage added in Phase 3; here we check values, consistency, and pre-calc
// hoisting directly.
//
#include "derivative_test_harness.hpp"
#include "eoslab/core/concepts.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/core/multifluid_base.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/residual_models/no_residual.hpp"

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
// to have linkage, which an anonymous-namespace type would not.
namespace mf_test {

// Relative-closeness check with an informative message.
void check_close(std::string_view name, double actual, double expected, double rtol = 1e-12)
{
    const double denom = std::max(1.0, std::abs(expected));
    expect(std::abs(actual - expected) / denom <= rtol) << name << ": actual=" << actual << " expected=" << expected;
}

struct IdealParams {
    double h;
    double g;
};

template<std::size_t N>
class MfIdealGas : public ge::BaseIdealEoS, public ge::MultiFluidBase<MfIdealGas<N>, IdealParams, N> {
public:
    using Base = ge::MultiFluidBase<MfIdealGas<N>, IdealParams, N>;
    using Base::Base;

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_i(Number c, const Number* x, Number T, std::size_t i,
                                          const IdealParams& prm) const
    {
        const Number R = ge::ideal_gas_constant<Number>;
        const Number phi = Number(prm.h) - (Number(prm.g) * std::log(T));
        return R * T * x[i] * (std::log(c) + std::log(x[i]) + phi);
    }

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density_i(const Number* rho_i, Number T, std::size_t i,
                                                  const IdealParams& prm) const
    {
        const Number R = ge::ideal_gas_constant<Number>;
        const Number phi = Number(prm.h) - (Number(prm.g) * std::log(T));
        return R * T * rho_i[i] * (std::log(rho_i[i]) + phi);
    }
};

template<class Number> struct LogTPre {
    Number logT;
};

template<std::size_t N>
class MfIdealGasPre : public ge::BaseIdealEoS, public ge::MultiFluidBase<MfIdealGasPre<N>, IdealParams, N> {
public:
    using Base = ge::MultiFluidBase<MfIdealGasPre<N>, IdealParams, N>;
    using Base::Base;

    inline static std::size_t precalc_calls = 0;

    template<std::floating_point Number>
    [[nodiscard]] LogTPre<Number> perform_pre_calculations(Number /*c*/, const Number* /*x*/, Number T) const
    {
        ++precalc_calls;
        return LogTPre<Number>{std::log(T)};
    }

    template<std::floating_point Number>
    [[nodiscard]] LogTPre<Number> perform_pre_calculations(const Number* /*rho_i*/, Number T) const
    {
        ++precalc_calls;
        return LogTPre<Number>{std::log(T)};
    }

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_i(Number c, const Number* x, Number T, std::size_t i, const IdealParams& prm,
                                          const LogTPre<Number>& pre) const
    {
        const Number R = ge::ideal_gas_constant<Number>;
        const Number phi = Number(prm.h) - (Number(prm.g) * pre.logT);
        return R * T * x[i] * (std::log(c) + std::log(x[i]) + phi);
    }

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density_i(const Number* rho_i, Number T, std::size_t i, const IdealParams& prm,
                                                  const LogTPre<Number>& pre) const
    {
        const Number R = ge::ideal_gas_constant<Number>;
        const Number phi = Number(prm.h) - (Number(prm.g) * pre.logT);
        return R * T * rho_i[i] * (std::log(rho_i[i]) + phi);
    }
};

// Independent analytic reference for the ideal-gas molar Helmholtz energy.
double ideal_a(double c, std::span<const double> x, double T, std::span<const IdealParams> prm)
{
    const double R = ge::ideal_gas_constant<double>;
    double mix = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double phi = prm[i].h - (prm[i].g * std::log(T));
        mix += x[i] * (std::log(x[i]) + phi);
    }
    return R * T * (std::log(c) + mix);
}

constexpr std::array<IdealParams, 2> kParams{IdealParams{2.5, 1.5}, IdealParams{3.1, 2.0}};

} // namespace mf_test

using namespace mf_test;

// Compile-time: the orchestrated models model the expected concepts.
static_assert(ge::EquationOfState<MfIdealGas<2>>);
static_assert(ge::IdealEoS<MfIdealGas<2>>);
static_assert(ge::EquationOfState<MfIdealGasPre<2>>);
static_assert(ge::IdealEoS<MfIdealGasPre<2>>);

int main()
{
    suite<"multifluid_base"> s = [] {
        const std::array<double, 2> x{0.4, 0.6};
        const std::span<const double> xs{x};
        const std::span<const IdealParams> prm{kParams};

        // ----------------------------------------------------------------- //
        // No-pre-calc model reproduces the analytic ideal-gas Helmholtz.
        // ----------------------------------------------------------------- //
        "no-precalc calc_helmholtz matches analytic"_test = [&] {
            const MfIdealGas<2> model{kParams};
            for (const double c : {50.0, 150.0, 400.0}) {
                for (const double T : {280.0, 350.0}) {
                    check_close("a", model.calc_helmholtz(c, x.data(), T), ideal_a(c, xs, T, prm));
                }
            }
        };

        // ----------------------------------------------------------------- //
        // Psi consistency: Psi == sum_i Psi_i and Psi == c * a.
        // ----------------------------------------------------------------- //
        "helmholtz_density consistency"_test = [&] {
            const MfIdealGas<2> model{kParams};
            const double c = 150.0;
            const double T = 320.0;
            const std::array<double, 2> rho{x[0] * c, x[1] * c};

            const double psi = model.calc_helmholtz_density(rho.data(), T);

            std::array<double, 2> psi_i{};
            model.calc_partial_helmholtz(rho.data(), T, psi_i.data());
            check_close("Psi == sum Psi_i", psi, psi_i[0] + psi_i[1]);
            check_close("Psi == c*a", psi, c * model.calc_helmholtz(c, x.data(), T));
        };

        // ----------------------------------------------------------------- //
        // The pre-calc model agrees with the inline (no-pre-calc) model.
        // ----------------------------------------------------------------- //
        "precalc model matches no-precalc model"_test = [&] {
            const MfIdealGas<2> plain{kParams};
            const MfIdealGasPre<2> pre{kParams};
            const double c = 200.0;
            const double T = 333.0;
            check_close("a", pre.calc_helmholtz(c, x.data(), T), plain.calc_helmholtz(c, x.data(), T));

            const std::array<double, 2> rho{x[0] * c, x[1] * c};
            check_close("Psi", pre.calc_helmholtz_density(rho.data(), T), plain.calc_helmholtz_density(rho.data(), T));
        };

        // ----------------------------------------------------------------- //
        // Pre-calc is hoisted: it runs exactly once per call, regardless of N.
        // ----------------------------------------------------------------- //
        "precalc runs once per call regardless of N"_test = [&] {
            const std::array<IdealParams, 3> p3{IdealParams{2.5, 1.5}, IdealParams{3.1, 2.0}, IdealParams{1.8, 1.1}};
            const MfIdealGasPre<3> model{p3};
            const std::array<double, 3> x3{0.2, 0.3, 0.5};

            MfIdealGasPre<3>::precalc_calls = 0;
            (void)model.calc_helmholtz(120.0, x3.data(), 300.0);
            expect(eq(MfIdealGasPre<3>::precalc_calls, std::size_t{1}));

            const std::array<double, 3> rho{24.0, 36.0, 60.0};
            MfIdealGasPre<3>::precalc_calls = 0;
            (void)model.calc_helmholtz_density(rho.data(), 300.0);
            expect(eq(MfIdealGasPre<3>::precalc_calls, std::size_t{1}));
        };

        // ----------------------------------------------------------------- //
        // Dynamic-N agrees with fixed-N.
        // ----------------------------------------------------------------- //
        "dynamic-N matches fixed-N"_test = [&] {
            const MfIdealGas<2> fixed{kParams};
            const std::vector<IdealParams> rows{kParams[0], kParams[1]};
            const MfIdealGas<std::dynamic_extent> dyn{std::span<const IdealParams>{rows}};

            const double c = 175.0;
            const double T = 310.0;
            expect(eq(dyn.size(), std::size_t{2}));
            check_close("a", dyn.calc_helmholtz(c, x.data(), T), fixed.calc_helmholtz(c, x.data(), T));
        };

        // ----------------------------------------------------------------- //
        // Enzyme forward + reverse mode differentiate correctly THROUGH the
        // CRTP base (pre-calc hoisting, for_each_component, get_parameters all
        // inlined). Paired with a zero residual, every derivative-based property
        // reduces to the analytic ideal-gas reference.
        // ----------------------------------------------------------------- //
        "enzyme derivative consistency through the CRTP base"_test = [] {
            const MfIdealGas<2> ideal{kParams};
            const ge::NoResidual<2> residual{};
            const ge::EoS<MfIdealGas<2>, ge::NoResidual<2>> eos{ideal, residual};

            eoslab_test::run_derivative_consistency_tests<2>(eos, 100.0, {0.4, 0.6}, 300.0);
            eoslab_test::run_derivative_consistency_tests<2>(eos, 250.0, {0.7, 0.3}, 350.0);
        };
    };
}
