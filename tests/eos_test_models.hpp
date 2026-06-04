#pragma once
//
// Analytic equation-of-state models used by the test-suite.
//
// These are deliberately simple, fully analytic models whose thermodynamics can
// be derived by hand. They are *not* meant to be physically accurate; they only
// have to be smooth, differentiable, and satisfy the IdealEoS / ResidualEoS
// concepts so that the core calculations (and the Enzyme autodiff that backs
// them) can be exercised and checked against closed-form references.
//
// Unit convention (matches the rest of the library):
//   c, rho_i : molar / partial-molar concentration  [mol / m^3]
//   x        : mole fraction                          [-]
//   T        : temperature                            [K]
//   helmholtz members return molar / volumetric Helmholtz energy:
//       calc_helmholtz          -> molar Helmholtz energy a            [J / mol]
//       calc_partial_helmholtz  -> per-component Helmholtz density Psi_i [J / m^3]
//   The two are consistent:  Psi(rho,T) = c * a(c, x, T)  with c = sum rho_i,
//   x_i = rho_i / c.
//
#include "eoslab/core/eos_base.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/core/numbers.hpp"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace eoslab_test {

// ---------------------------------------------------------------------------
// Ideal-gas model
//
//   a_id(c, x, T) = R T [ ln c + sum_i x_i ln x_i + sum_i x_i phi_i(T) ]
//   phi_i(T)      = h_i - g_i ln T
//
// The c-dependence is exactly R T ln c, i.e. d a_id / d c = R T / c, which is
// the defining property of an ideal gas (compressibility factor Z = 1). The
// h_i, g_i constants give the model an arbitrary-but-smooth caloric behaviour
// so that internal energy / entropy / cv have something non-trivial to test.
//
//   Psi_id_i(rho, T) = R T rho_i [ ln rho_i + phi_i(T) ]
// ---------------------------------------------------------------------------
template<std::size_t N> class IdealGasTestModel : public glis::eos::BaseIdealEoS, public glis::eos::BaseEoS<N> {
public:
    std::array<double, N> h{}; // dimensionless ideal-reference enthalpy-like constants
    std::array<double, N> g{}; // dimensionless ideal-reference heat-capacity-like constants

    constexpr IdealGasTestModel() = default;
    constexpr IdealGasTestModel(std::array<double, N> h_in, std::array<double, N> g_in) : h{h_in}, g{g_in} {}

    using glis::eos::BaseEoS<N>::size;

    template<std::floating_point Number> [[nodiscard]] Number calc_helmholtz(Number c, const Number* x, Number T) const
    {
        const Number R = glis::eos::ideal_gas_constant<Number>;
        Number mix{0};
        for (std::size_t i = 0; i < N; ++i) {
            const Number phi = Number(h[i]) - (Number(g[i]) * std::log(T));
            mix += x[i] * (std::log(x[i]) + phi);
        }
        return R * T * (std::log(c) + mix);
    }

    template<std::floating_point Number> void calc_partial_helmholtz(const Number* rho, Number T, Number* out) const
    {
        const Number R = glis::eos::ideal_gas_constant<Number>;
        for (std::size_t i = 0; i < N; ++i) {
            const Number phi = Number(h[i]) - (Number(g[i]) * std::log(T));
            out[i] = R * T * rho[i] * (std::log(rho[i]) + phi);
        }
    }
};

// ---------------------------------------------------------------------------
// Residual model (truncated virial form)
//
//   alpha^r(c, x, T) = B(T, x) c + (1/2) C(T, x) c^2
//   a_res(c, x, T)   = R T alpha^r = R T [ B c + (1/2) C c^2 ]
//   B(T, x) = sum_i x_i (b_i - beta_i / T)        [m^3 / mol]
//   C(T, x) = sum_i x_i gamma_i                   [m^6 / mol^2]
//
// This yields the virial pressure  p = R T (c + B c^2 + C c^3).
//
// Volumetric Helmholtz density (function of rho, T only):
//   with c = sum rho_i, S1 = sum rho_i (b_i - beta_i/T) = B c,
//        S2 = sum rho_i gamma_i = C c
//   Psi_res(rho, T) = c a_res = R T [ c S1 + (1/2) c^2 S2 ]
// split per component (any split summing to Psi_res is valid for mu_i):
//   Psi_res_i = R T [ rho_i S1 + (1/2) rho_i c S2 ]
// ---------------------------------------------------------------------------
template<std::size_t N> class VirialResidualTestModel : public glis::eos::BaseEoS<N> {
public:
    std::array<double, N> b{};     // [m^3 / mol]
    std::array<double, N> beta{};  // [K m^3 / mol]
    std::array<double, N> gamma{}; // [m^6 / mol^2]

    constexpr VirialResidualTestModel() = default;
    constexpr VirialResidualTestModel(std::array<double, N> b_in, std::array<double, N> beta_in,
                                      std::array<double, N> gamma_in) :
        b{b_in}, beta{beta_in}, gamma{gamma_in}
    {
    }

    using glis::eos::BaseEoS<N>::size;

    template<std::floating_point Number> [[nodiscard]] Number calc_helmholtz(Number c, const Number* x, Number T) const
    {
        const Number R = glis::eos::ideal_gas_constant<Number>;
        Number B{0};
        Number C{0};
        for (std::size_t i = 0; i < N; ++i) {
            B += x[i] * (Number(b[i]) - (Number(beta[i]) / T));
            C += x[i] * Number(gamma[i]);
        }
        return R * T * ((B * c) + (Number{0.5} * C * c * c));
    }

    template<std::floating_point Number> void calc_partial_helmholtz(const Number* rho, Number T, Number* out) const
    {
        const Number R = glis::eos::ideal_gas_constant<Number>;
        Number c{0};
        Number S1{0};
        Number S2{0};
        for (std::size_t i = 0; i < N; ++i) {
            c += rho[i];
            S1 += rho[i] * (Number(b[i]) - (Number(beta[i]) / T));
            S2 += rho[i] * Number(gamma[i]);
        }
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = R * T * ((rho[i] * S1) + (Number{0.5} * rho[i] * c * S2));
        }
    }
};

// ---------------------------------------------------------------------------
// Convenience factory: a representative two-component model pair.
// ---------------------------------------------------------------------------
inline auto make_binary_model()
{
    IdealGasTestModel<2> ideal{{2.5, 3.1}, {1.5, 2.0}};
    VirialResidualTestModel<2> residual{{1.0e-3, 1.5e-3}, {1.0e-1, 8.0e-2}, {2.0e-6, 1.0e-6}};
    return glis::eos::EoS<IdealGasTestModel<2>, VirialResidualTestModel<2>>{ideal, residual};
}

inline auto make_unary_model()
{
    IdealGasTestModel<1> ideal{{2.5}, {1.5}};
    VirialResidualTestModel<1> residual{{1.0e-3}, {1.0e-1}, {2.0e-6}};
    return glis::eos::EoS<IdealGasTestModel<1>, VirialResidualTestModel<1>>{ideal, residual};
}

} // namespace eoslab_test
