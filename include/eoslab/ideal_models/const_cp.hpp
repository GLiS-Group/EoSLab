#pragma once
/**
 * @file const_cp.hpp
 * @brief Ideal-gas model in which every pure species has a constant isobaric
 *        molar heat capacity @f$c_p@f$.
 */

#include "eoslab/core/concepts.hpp"
#include "eoslab/core/eos_base.hpp"
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
 * @brief Ideal-gas equation of state with a constant isobaric molar heat
 *        capacity per species.
 *
 * With @f$c_p@f$ constant the molar enthalpy and entropy of each pure species
 * are closed-form functions of temperature, referenced to a per-species
 * @f$(T_\mathrm{ref}, p_\mathrm{ref})@f$ state. The mixture's ideal Helmholtz
 * energy is the sum of those per-species contributions plus the ideal entropy
 * of mixing.
 *
 * Construct the model from one SpeciesInput per species (each species may use
 * its own reference state), pair it with a residual model in an EoS, and
 * evaluate properties through the free functions in core_calculations.hpp:
 *
 * @code{.cpp}
 * using namespace glis::eos;
 *
 * ConstantCp<2> ideal(std::array{
 *     ConstantCp<2>::SpeciesInput{
 *         .T_ref = 298.15, .p_ref = 1.0e5, .c_p = 29.1, .h_ref = 0.0, .s_ref = 191.6},
 *     ConstantCp<2>::SpeciesInput{
 *         .T_ref = 298.15, .p_ref = 1.0e5, .c_p = 33.6, .h_ref = 0.0, .s_ref = 205.2},
 * });
 * EoS eos{ideal, NoResidual<2>{}};
 *
 * const std::array<double, 2> x{0.5, 0.5}; // mole fractions
 * const double c = 40.0;                   // molar concentration [mol/m^3]
 * const double T = 350.0;                  // temperature [K]
 * const double h = calc_enthalpy(eos, c, std::span<const double, 2>{x}, T);
 * @endcode
 *
 * Use the @c std::dynamic_extent default (e.g. `ConstantCp<>`) when the number
 * of species is only known at run time:
 *
 * @code{.cpp}
 * std::vector<ConstantCp<>::SpeciesInput> inputs = load_species(...);
 * ConstantCp<> ideal{std::span<const ConstantCp<>::SpeciesInput>{inputs}};
 * @endcode
 *
 * @tparam N Component count, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent> class ConstantCp : public BaseEoS<N>, public BaseIdealEoS {
public:
    /**
     * @brief Natural per-species reference data supplied by the user.
     *
     * The constructor turns these into the derived parameters the model stores.
     * The reference molar concentration is *not* supplied directly; it is
     * computed as @f$c_\mathrm{ref} = p_\mathrm{ref} / (R\,T_\mathrm{ref})@f$.
     */
    struct SpeciesInput {
        double T_ref; ///< Reference temperature @f$T_\mathrm{ref}@f$ [K].
        double p_ref; ///< Reference pressure @f$p_\mathrm{ref}@f$ [Pa].
        double c_p;   ///< Isobaric molar heat capacity @f$c_p@f$ [J/(mol K)].
        double h_ref; ///< Reference molar enthalpy @f$h_\mathrm{ref}@f$ [J/mol].
        double s_ref; ///< Reference molar entropy @f$s_\mathrm{ref}@f$ [J/(mol K)].
    };

    /**
     * @brief Construct a compile-time-sized model from per-species reference data.
     *
     * Only available when the component count @p N is known at compile time.
     *
     * @param inputs One SpeciesInput per species.
     */
    explicit ConstantCp(const std::array<SpeciesInput, N>& inputs)
        requires(N != std::dynamic_extent)
    {
        for (std::size_t i = 0; i < N; ++i) {
            scatter(i, N, inputs[i]);
        }
    }

    /**
     * @brief Construct a runtime-sized model from per-species reference data.
     *
     * Only available when @p N is @c std::dynamic_extent; `size()` becomes
     * `inputs.size()`.
     *
     * @param inputs One SpeciesInput per species.
     */
    explicit ConstantCp(std::span<const SpeciesInput> inputs)
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
        // Pre-calculations hoisted out of the component loop.
        const Number TlnT = T * std::log(T);
        const Number RTlnC = R * T * std::log(c);

        Number a{0};
        for (std::size_t i = 0; i < n; ++i) {
            const double T_ref = data_[(col_T_ref * n) + i];
            const double R_ln_c_ref = data_[(col_R_ln_c_ref * n) + i];
            const double c_p = data_[(col_c_p * n) + i];
            const double ln_T_ref = data_[(col_ln_T_ref * n) + i];
            const double h_ref = data_[(col_h_ref * n) + i];
            const double s_ref = data_[(col_s_ref * n) + i];
            a += (Number{2} * R * T * xlnx<0>(x[i])) +
                 (x[i] * (h_ref + (c_p * (T - T_ref)) - (T * s_ref) + (c_p * T * ln_T_ref) + RTlnC - (T * R_ln_c_ref) +
                          ((R - c_p) * TlnT) - (R * T * ln_T_ref) - (R * T)));
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
        // c = sum_i rho_i, then the hoisted pre-calculations.
        Number c{0};
        for (std::size_t i = 0; i < n; ++i) {
            c += rho_i[i];
        }
        const Number TlnT = T * std::log(T);
        const Number RTlnC = R * T * std::log(c);

        Number psi{0};
        for (std::size_t i = 0; i < n; ++i) {
            const double T_ref = data_[(col_T_ref * n) + i];
            const double R_ln_c_ref = data_[(col_R_ln_c_ref * n) + i];
            const double c_p = data_[(col_c_p * n) + i];
            const double ln_T_ref = data_[(col_ln_T_ref * n) + i];
            const double h_ref = data_[(col_h_ref * n) + i];
            const double s_ref = data_[(col_s_ref * n) + i];
            psi += (Number{2} * R * T * xlnx(rho_i[i])) +
                   (rho_i[i] * (h_ref + (c_p * (T - T_ref)) - (T * s_ref) + (c_p * T * ln_T_ref) - RTlnC -
                                (T * R_ln_c_ref) + ((R - c_p) * TlnT) - (R * T * ln_T_ref) - (R * T)));
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
        const Number TlnT = T * std::log(T);
        const Number RTlnC = R * T * std::log(c);

        for (std::size_t i = 0; i < n; ++i) {
            const double T_ref = data_[(col_T_ref * n) + i];
            const double R_ln_c_ref = data_[(col_R_ln_c_ref * n) + i];
            const double c_p = data_[(col_c_p * n) + i];
            const double ln_T_ref = data_[(col_ln_T_ref * n) + i];
            const double h_ref = data_[(col_h_ref * n) + i];
            const double s_ref = data_[(col_s_ref * n) + i];
            out[i] = (Number{2} * R * T * xlnx(rho_i[i])) +
                     (rho_i[i] * (h_ref + (c_p * (T - T_ref)) - (T * s_ref) + (c_p * T * ln_T_ref) - RTlnC -
                                  (T * R_ln_c_ref) + ((R - c_p) * TlnT) - (R * T * ln_T_ref) - (R * T)));
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

    // Column layout of the SoA storage.
    static constexpr std::size_t col_T_ref = 0;      // T_ref [K]
    static constexpr std::size_t col_R_ln_c_ref = 1; // R ln(c_ref) [J/(mol K)]
    static constexpr std::size_t col_c_p = 2;        // c_p [J/(mol K)]
    static constexpr std::size_t col_ln_T_ref = 3;   // ln(T_ref) [-]
    static constexpr std::size_t col_h_ref = 4;      // h_ref [J/mol]
    static constexpr std::size_t col_s_ref = 5;      // s_ref [J/(mol K)]
    static constexpr std::size_t num_params = 6;

    using Storage =
        std::conditional_t<N == std::dynamic_extent, std::vector<double>, std::array<double, num_params * N>>;

    // Derive species i's parameters from `in` and write them into the columns.
    void scatter(std::size_t i, std::size_t n, const SpeciesInput& in)
    {
        constexpr double R = ideal_gas_constant<double>;
        const double c_ref = in.p_ref / (R * in.T_ref);
        data_[(col_T_ref * n) + i] = in.T_ref;
        data_[(col_R_ln_c_ref * n) + i] = R * std::log(c_ref);
        data_[(col_c_p * n) + i] = in.c_p;
        data_[(col_ln_T_ref * n) + i] = std::log(in.T_ref);
        data_[(col_h_ref * n) + i] = in.h_ref;
        data_[(col_s_ref * n) + i] = in.s_ref;
    }

    Storage data_{}; // Column-major (SoA) parameter storage.
};

static_assert(IdealEoS<ConstantCp<2>>, "ConstantCp must satisfy the IdealEoS concept.");
static_assert(IdealEoS<ConstantCp<std::dynamic_extent>>, "ConstantCp must satisfy the IdealEoS concept.");
} // namespace glis::eos
