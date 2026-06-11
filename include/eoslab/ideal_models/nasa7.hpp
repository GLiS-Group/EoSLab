#pragma once
/**
 * @file nasa7.hpp
 * @brief Ideal-gas model using the 7-coefficient NASA polynomial
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
 * @brief Ideal-gas equation of state parameterized by the NASA-7 polynomials.
 *
 * Each species' standard-state heat capacity, enthalpy, and entropy follow the
 * 7-coefficient NASA polynomial form
 * @f[
 *   c_p(T)/R = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4,
 * @f]
 * with @f$a_5@f$ and @f$a_6@f$ setting the enthalpy and entropy references.
 * The mixture's ideal Helmholtz energy is the sum of the per-species
 * contributions plus the ideal entropy of mixing.
 *
 * Construct the model from one SpeciesInput per species (coefficients plus a
 * reference state, which may differ per species), pair it with a residual
 * model in an EoS, and evaluate properties through the free functions in
 * core_calculations.hpp:
 *
 * @code{.cpp}
 * using namespace glis::eos;
 *
 * // N2, low-temperature (200-1000 K) NASA-7 coefficients.
 * Nasa7<1> ideal(std::array{Nasa7<1>::SpeciesInput{
 *     .a0 = 3.53100528, .a1 = -1.23660988e-4, .a2 = -5.02999433e-7,
 *     .a3 = 2.43530612e-9, .a4 = -1.40881235e-12, .a5 = -1046.97628,
 *     .a6 = 2.96747038, .T_ref = 298.15, .p_ref = 1.0e5}});
 * EoS eos{ideal, NoResidual<1>{}};
 *
 * const std::array<double, 1> x{1.0}; // mole fractions
 * const double c = 40.0;              // molar concentration [mol/m^3]
 * const double T = 350.0;             // temperature [K]
 * const double cp = calc_cp(eos, c, std::span<const double, 1>{x}, T);
 * @endcode
 *
 * Use the @c std::dynamic_extent default (e.g. `Nasa7<>`) when the number of
 * species is only known at run time:
 *
 * @code{.cpp}
 * std::vector<Nasa7<>::SpeciesInput> inputs = load_species(...);
 * Nasa7<> ideal{std::span<const Nasa7<>::SpeciesInput>{inputs}};
 * @endcode
 *
 * @tparam N Component count, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent> class Nasa7 : public BaseEoS<N>, public BaseIdealEoS {
public:
    /// @brief Natural per-species NASA-7 coefficients and reference state.
    struct SpeciesInput {
        double a0;    ///< NASA-7 coefficient @f$a_0@f$.
        double a1;    ///< NASA-7 coefficient @f$a_1@f$.
        double a2;    ///< NASA-7 coefficient @f$a_2@f$.
        double a3;    ///< NASA-7 coefficient @f$a_3@f$.
        double a4;    ///< NASA-7 coefficient @f$a_4@f$.
        double a5;    ///< NASA-7 coefficient @f$a_5@f$ (enthalpy reference).
        double a6;    ///< NASA-7 coefficient @f$a_6@f$ (entropy reference).
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
    explicit Nasa7(const std::array<SpeciesInput, N>& inputs)
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
    explicit Nasa7(std::span<const SpeciesInput> inputs)
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

        Number a{0};
        for (std::size_t i = 0; i < n; ++i) {
            // The NASA-7 Helmholtz contribution is a 5th-degree polynomial in T.
            std::array<Number, 6> coeffs;
            coeffs[0] = data_[(col_a5 * n) + i];
            coeffs[1] = (data_[(col_a0 * n) + i] * (Number{1} - lnT)) - data_[(col_a6 * n) + i] + lnT + lnC -
                        data_[(col_ln_cref_Tref * n) + i] - Number{1};
            coeffs[2] = -data_[(col_a1_over_2 * n) + i];
            coeffs[3] = -data_[(col_a2_over_6 * n) + i];
            coeffs[4] = -data_[(col_a3_over_12 * n) + i];
            coeffs[5] = -data_[(col_a4_over_20 * n) + i];
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

        Number psi{0};
        for (std::size_t i = 0; i < n; ++i) {
            std::array<Number, 6> coeffs;
            coeffs[0] = data_[(col_a5 * n) + i];
            coeffs[1] = (data_[(col_a0 * n) + i] * (Number{1} - lnT)) - data_[(col_a6 * n) + i] + lnT - lnC -
                        data_[(col_ln_cref_Tref * n) + i] - Number{1};
            coeffs[2] = -data_[(col_a1_over_2 * n) + i];
            coeffs[3] = -data_[(col_a2_over_6 * n) + i];
            coeffs[4] = -data_[(col_a3_over_12 * n) + i];
            coeffs[5] = -data_[(col_a4_over_20 * n) + i];
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

        for (std::size_t i = 0; i < n; ++i) {
            std::array<Number, 6> coeffs;
            coeffs[0] = data_[(col_a5 * n) + i];
            coeffs[1] = (data_[(col_a0 * n) + i] * (Number{1} - lnT)) - data_[(col_a6 * n) + i] + lnT - lnC -
                        data_[(col_ln_cref_Tref * n) + i] - Number{1};
            coeffs[2] = -data_[(col_a1_over_2 * n) + i];
            coeffs[3] = -data_[(col_a2_over_6 * n) + i];
            coeffs[4] = -data_[(col_a3_over_12 * n) + i];
            coeffs[5] = -data_[(col_a4_over_20 * n) + i];
            out[i] = R * ((rho_i[i] * eval_polynomial<5>(coeffs, T)) + (T * Number{2} * xlnx<0>(rho_i[i])));
        }
    }

private:
    // Implementation notes (deliberate, benchmarked choices):
    //  * The kernels are written out in full -- no CRTP indirection, no
    //    parameter-struct reflection -- so the expression Enzyme differentiates
    //    stays flat and the autodiff is robust at every optimization level.
    //  * The derived per-species parameters are stored Structure-of-Arrays
    //    (column-major: one contiguous column per parameter across all
    //    species), which benchmarked faster than an Array-of-Structs layout.
    //  * The a1..a4 coefficients are stored pre-divided by the constants the
    //    Helmholtz polynomial needs, so the kernels only negate them.

    // Column layout of the SoA storage.
    static constexpr std::size_t col_a0 = 0;
    static constexpr std::size_t col_a1_over_2 = 1;
    static constexpr std::size_t col_a2_over_6 = 2;
    static constexpr std::size_t col_a3_over_12 = 3;
    static constexpr std::size_t col_a4_over_20 = 4;
    static constexpr std::size_t col_a5 = 5;
    static constexpr std::size_t col_a6 = 6;
    static constexpr std::size_t col_ln_cref_Tref = 7; // ln(c_ref * T_ref)
    static constexpr std::size_t num_params = 8;

    using Storage =
        std::conditional_t<N == std::dynamic_extent, std::vector<double>, std::array<double, num_params * N>>;

    // Derive species i's parameters from `in` and write them into the columns.
    void scatter(std::size_t i, std::size_t n, const SpeciesInput& in)
    {
        constexpr double R = ideal_gas_constant<double>;
        const double c_ref = in.p_ref / (R * in.T_ref);
        data_[(col_a0 * n) + i] = in.a0;
        data_[(col_a1_over_2 * n) + i] = in.a1 / 2.;
        data_[(col_a2_over_6 * n) + i] = in.a2 / 6.;
        data_[(col_a3_over_12 * n) + i] = in.a3 / 12.;
        data_[(col_a4_over_20 * n) + i] = in.a4 / 20.;
        data_[(col_a5 * n) + i] = in.a5;
        data_[(col_a6 * n) + i] = in.a6;
        data_[(col_ln_cref_Tref * n) + i] = std::log(c_ref * in.T_ref);
    }

    Storage data_{}; // Column-major (SoA) parameter storage.
};

static_assert(IdealEoS<Nasa7<2>>, "Nasa7 must satisfy the IdealEoS concept.");
static_assert(IdealEoS<Nasa7<std::dynamic_extent>>, "Nasa7 must satisfy the IdealEoS concept.");
} // namespace glis::eos
