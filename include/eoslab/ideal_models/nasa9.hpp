#pragma once
/**
 * @file nasa9.hpp
 * @brief Ideal-gas model using the 9-coefficient NASA polynomial
 *        parameterization of the per-species standard-state thermodynamics.
 */

#include "eoslab/core/concepts.hpp"
#include "eoslab/core/eos_base.hpp"
#include "eoslab/core/horner.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/core/xlnx.hpp"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace glis::eos {

/**
 * @brief Ideal-gas equation of state parameterized by the NASA-9 polynomials.
 *
 * Each species' standard-state heat capacity, enthalpy, and entropy follow the
 * 9-coefficient NASA (Glenn) polynomial form
 * @f[
 *   c_p(T)/R = a_0 T^{-2} + a_1 T^{-1} + a_2 + a_3 T + a_4 T^2 + a_5 T^3 + a_6 T^4,
 * @f]
 * with @f$a_7@f$ and @f$a_8@f$ setting the enthalpy and entropy references.
 * The two extra inverse-temperature terms are what distinguish NASA-9 from
 * NASA-7; with @f$a_0 = a_1 = 0@f$ this model coincides exactly with Nasa7
 * using the coefficients @f$a_2 \dots a_8@f$. The mixture's ideal Helmholtz
 * energy is the sum of the per-species contributions plus the ideal entropy
 * of mixing.
 *
 * Construct the model from one SpeciesInput per species (coefficients plus a
 * reference state, which may differ per species), pair it with a residual
 * model in an EoS, and evaluate properties through the free functions in
 * core_calculations.hpp:
 *
 * @code{.cpp}
 * using namespace glis::eos;
 *
 * // N2, low-temperature (200-1000 K) NASA-9 coefficients.
 * Nasa9<1> ideal(std::array{Nasa9<1>::SpeciesInput{
 *     .a0 = 2.210371497e4, .a1 = -3.818461820e2, .a2 = 6.082738360,
 *     .a3 = -8.530914410e-3, .a4 = 1.384646189e-5, .a5 = -9.625793620e-9,
 *     .a6 = 2.519705809e-12, .a7 = 7.108460860e2, .a8 = -1.076003744e1,
 *     .T_ref = 298.15, .p_ref = 1.0e5}});
 * EoS eos{ideal, NoResidual<1>{}};
 *
 * const std::array<double, 1> x{1.0}; // mole fractions
 * const double c = 40.0;              // molar concentration [mol/m^3]
 * const double T = 350.0;             // temperature [K]
 * const double cp = calc_cp(eos, c, std::span<const double, 1>{x}, T);
 * @endcode
 *
 * Use the @c std::dynamic_extent default (e.g. `Nasa9<>`) when the number of
 * species is only known at run time:
 *
 * @code{.cpp}
 * std::vector<Nasa9<>::SpeciesInput> inputs = load_species(...);
 * Nasa9<> ideal{std::span<const Nasa9<>::SpeciesInput>{inputs}};
 * @endcode
 *
 * @tparam N Component count, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent> class Nasa9 : public BaseEoS<N>, public BaseIdealEoS {
public:
    /// @brief Natural per-species NASA-9 coefficients and reference state.
    struct SpeciesInput {
        double a0;    ///< NASA-9 coefficient @f$a_0@f$ (the @f$T^{-2}@f$ term).
        double a1;    ///< NASA-9 coefficient @f$a_1@f$ (the @f$T^{-1}@f$ term).
        double a2;    ///< NASA-9 coefficient @f$a_2@f$.
        double a3;    ///< NASA-9 coefficient @f$a_3@f$.
        double a4;    ///< NASA-9 coefficient @f$a_4@f$.
        double a5;    ///< NASA-9 coefficient @f$a_5@f$.
        double a6;    ///< NASA-9 coefficient @f$a_6@f$.
        double a7;    ///< NASA-9 coefficient @f$a_7@f$ (enthalpy reference).
        double a8;    ///< NASA-9 coefficient @f$a_8@f$ (entropy reference).
        double T_ref; ///< Reference temperature @f$T_\mathrm{ref}@f$ [K].
        double p_ref; ///< Reference pressure @f$p_\mathrm{ref}@f$ [Pa].
    };

    /**
     * @brief Construct a compile-time-sized model from per-species inputs.
     *
     * Only available when the component count @p N is known at compile time.
     *
     * @param inputs One SpeciesInput per species.
     */
    explicit Nasa9(const std::array<SpeciesInput, N>& inputs)
        requires(N != std::dynamic_extent)
    {
        for (std::size_t i = 0; i < N; ++i) {
            scatter(i, N, inputs[i]);
        }
    }

    /**
     * @brief Construct a runtime-sized model from per-species inputs.
     *
     * Only available when @p N is @c std::dynamic_extent; `size()` becomes
     * `inputs.size()`.
     *
     * @param inputs One SpeciesInput per species.
     */
    explicit Nasa9(std::span<const SpeciesInput> inputs)
        requires(N == std::dynamic_extent)
        : BaseEoS<N>(inputs.size())
    {
        const std::size_t n = inputs.size();
        data_.resize(num_params * n);
        for (std::size_t i = 0; i < n; ++i) {
            scatter(i, n, inputs[i]);
        }
    }

    /**
     * @brief Total molar Helmholtz energy @f$a = \sum_i a_i@f$.
     * @param c Molar concentration [mol/m^3].
     * @param x Mole-fraction array [-].
     * @param T Temperature [K].
     * @return Molar Helmholtz energy [J/mol].
     */
    template<std::floating_point Number> [[nodiscard]] Number calc_helmholtz(Number c, const Number* x, Number T) const
    {
        const Number R = ideal_gas_constant<Number>;
        const std::size_t n = this->size();
        const Number lnC = std::log(c);
        const Number lnT = std::log(T);
        const Number one_over_T = Number{1} / T;

        Number a{0};
        for (std::size_t i = 0; i < n; ++i) {
            // The NASA-9 Helmholtz contribution is a 5th-degree polynomial in T
            // plus 1/T and lnT-dependent constants, folded into coeffs[0].
            std::array<Number, 6> coeffs;
            coeffs[0] = (data_[(col_a1 * n) + i] * (lnT + Number{1})) + data_[(col_a7 * n) + i] -
                        (data_[(col_a0_over_2 * n) + i] * one_over_T);
            coeffs[1] = (data_[(col_a2 * n) + i] * (Number{1} - lnT)) - data_[(col_a8 * n) + i] + lnT + lnC -
                        data_[(col_ln_cref_Tref * n) + i] - Number{1};
            coeffs[2] = -data_[(col_a3_over_2 * n) + i];
            coeffs[3] = -data_[(col_a4_over_6 * n) + i];
            coeffs[4] = -data_[(col_a5_over_12 * n) + i];
            coeffs[5] = -data_[(col_a6_over_20 * n) + i];
            a += R * ((x[i] * eval_polynomial<5>(coeffs, T)) + (T * Number{2} * xlnx<0>(x[i])));
        }
        return a;
    }

    /**
     * @brief Total Helmholtz energy density @f$\Psi = \sum_i \Psi_i@f$.
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @return Helmholtz energy density [J/m^3].
     */
    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density(const Number* rho_i, Number T) const
    {
        const Number R = ideal_gas_constant<Number>;
        const std::size_t n = this->size();
        Number c{0};
        for (std::size_t i = 0; i < n; ++i) {
            c += rho_i[i];
        }
        const Number lnC = std::log(c);
        const Number lnT = std::log(T);
        const Number one_over_T = Number{1} / T;

        Number psi{0};
        for (std::size_t i = 0; i < n; ++i) {
            std::array<Number, 6> coeffs;
            coeffs[0] = (data_[(col_a1 * n) + i] * (lnT + Number{1})) + data_[(col_a7 * n) + i] -
                        (data_[(col_a0_over_2 * n) + i] * one_over_T);
            coeffs[1] = (data_[(col_a2 * n) + i] * (Number{1} - lnT)) - data_[(col_a8 * n) + i] + lnT - lnC -
                        data_[(col_ln_cref_Tref * n) + i] - Number{1};
            coeffs[2] = -data_[(col_a3_over_2 * n) + i];
            coeffs[3] = -data_[(col_a4_over_6 * n) + i];
            coeffs[4] = -data_[(col_a5_over_12 * n) + i];
            coeffs[5] = -data_[(col_a6_over_20 * n) + i];
            psi += R * ((rho_i[i] * eval_polynomial<5>(coeffs, T)) + (T * Number{2} * xlnx<0>(rho_i[i])));
        }
        return psi;
    }

    /**
     * @brief Per-component Helmholtz energy density, @f$\text{out}[i] = \Psi_i@f$.
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @param[out] out Per-component Helmholtz energy density [J/m^3]; length `size()`.
     */
    template<std::floating_point Number> void calc_partial_helmholtz(const Number* rho_i, Number T, Number* out) const
    {
        const Number R = ideal_gas_constant<Number>;
        const std::size_t n = this->size();
        Number c{0};
        for (std::size_t i = 0; i < n; ++i) {
            c += rho_i[i];
        }
        const Number lnC = std::log(c);
        const Number lnT = std::log(T);
        const Number one_over_T = Number{1} / T;

        for (std::size_t i = 0; i < n; ++i) {
            std::array<Number, 6> coeffs;
            coeffs[0] = (data_[(col_a1 * n) + i] * (lnT + Number{1})) + data_[(col_a7 * n) + i] -
                        (data_[(col_a0_over_2 * n) + i] * one_over_T);
            coeffs[1] = (data_[(col_a2 * n) + i] * (Number{1} - lnT)) - data_[(col_a8 * n) + i] + lnT - lnC -
                        data_[(col_ln_cref_Tref * n) + i] - Number{1};
            coeffs[2] = -data_[(col_a3_over_2 * n) + i];
            coeffs[3] = -data_[(col_a4_over_6 * n) + i];
            coeffs[4] = -data_[(col_a5_over_12 * n) + i];
            coeffs[5] = -data_[(col_a6_over_20 * n) + i];
            out[i] = R * ((rho_i[i] * eval_polynomial<5>(coeffs, T)) + (T * Number{2} * xlnx<0>(rho_i[i])));
        }
    }

private:
    // Same deliberate choices as Nasa7 (see the notes there): flat kernels for
    // Enzyme, Structure-of-Arrays storage, coefficients pre-divided by the
    // constants the Helmholtz polynomial needs.

    // Column layout of the SoA storage.
    static constexpr std::size_t col_a0_over_2 = 0;
    static constexpr std::size_t col_a1 = 1;
    static constexpr std::size_t col_a2 = 2;
    static constexpr std::size_t col_a3_over_2 = 3;
    static constexpr std::size_t col_a4_over_6 = 4;
    static constexpr std::size_t col_a5_over_12 = 5;
    static constexpr std::size_t col_a6_over_20 = 6;
    static constexpr std::size_t col_a7 = 7;
    static constexpr std::size_t col_a8 = 8;
    static constexpr std::size_t col_ln_cref_Tref = 9; // ln(c_ref * T_ref)
    static constexpr std::size_t num_params = 10;

    using Storage =
        std::conditional_t<N == std::dynamic_extent, std::vector<double>, std::array<double, num_params * N>>;

    // Derive species i's parameters from `in` and write them into the columns.
    void scatter(std::size_t i, std::size_t n, const SpeciesInput& in)
    {
        constexpr double R = ideal_gas_constant<double>;
        const double c_ref = in.p_ref / (R * in.T_ref);
        data_[(col_a0_over_2 * n) + i] = in.a0 / 2.;
        data_[(col_a1 * n) + i] = in.a1;
        data_[(col_a2 * n) + i] = in.a2;
        data_[(col_a3_over_2 * n) + i] = in.a3 / 2.;
        data_[(col_a4_over_6 * n) + i] = in.a4 / 6.;
        data_[(col_a5_over_12 * n) + i] = in.a5 / 12.;
        data_[(col_a6_over_20 * n) + i] = in.a6 / 20.;
        data_[(col_a7 * n) + i] = in.a7;
        data_[(col_a8 * n) + i] = in.a8;
        data_[(col_ln_cref_Tref * n) + i] = std::log(c_ref * in.T_ref);
    }

    Storage data_{}; // Column-major (SoA) parameter storage.
};

static_assert(IdealEoS<Nasa9<2>>, "Nasa9 must satisfy the IdealEoS concept.");
static_assert(IdealEoS<Nasa9<std::dynamic_extent>>, "Nasa9 must satisfy the IdealEoS concept.");
} // namespace glis::eos
