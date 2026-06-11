#pragma once
/**
 * @file const_cp.hpp
 * @brief Ideal-gas model in which every pure species has a constant isobaric
 *        molar heat capacity @f$c_p@f$.
 *
 * With @f$c_p@f$ constant the molar enthalpy and entropy of each pure species are
 * closed-form functions of temperature, referenced to a per-species
 * @f$(T_\mathrm{ref}, p_\mathrm{ref})@f$ state. This model assembles the mixture's
 * ideal Helmholtz energy as a sum of those per-species contributions (plus the
 * ideal entropy of mixing).
 *
 * The model is deliberately *self-contained*: it derives directly from the EoS
 * base classes and implements `calc_helmholtz`, `calc_helmholtz_density`, and
 * `calc_partial_helmholtz` as plain functions whose pre-calculations and
 * component loop are written out in full. This keeps the expression that
 * [Enzyme](https://enzyme.mit.edu) differentiates as flat as possible &mdash; no
 * CRTP indirection, no parameter-storage reflection &mdash; which makes the
 * autodiff robust at every optimization level. Parameters are stored
 * Array-of-Structs (one @ref Params per species).
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
 * Construct it from a per-species list of natural reference data
 * (SpeciesInput); the constructor performs the pre-calculations and stores the
 * derived per-species parameters. Each species may use a different reference
 * temperature and pressure.
 *
 * @tparam N Component count, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent> class ConstantCp : public BaseEoS<N>, public BaseIdealEoS {
public:
    /**
     * @brief Natural per-species reference data supplied by the user.
     *
     * The constructor turns these into the stored @ref Params. The reference
     * molar concentration is *not* supplied directly; it is computed as
     * @f$c_\mathrm{ref} = p_\mathrm{ref} / (R\,T_\mathrm{ref})@f$.
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
            params_[i] = to_params(inputs[i]);
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
        params_.reserve(inputs.size());
        for (const SpeciesInput& in : inputs) {
            params_.push_back(to_params(in));
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
        // Pre-calculations hoisted out of the component loop.
        const Number TlnT = T * std::log(T);
        const Number RTlnC = R * T * std::log(c);

        Number a{0};
        for (std::size_t i = 0; i < this->size(); ++i) {
            const Params& p = params_[i];
            a += (Number{2} * R * T * xlnx<0>(x[i])) +
                 (x[i] * (p.h_ref + (p.c_p * (T - p.T_ref)) - (T * p.s_ref) + (p.c_p * T * p.ln_T_ref) + RTlnC -
                          (T * p.R_ln_c_ref) + ((R - p.c_p) * TlnT) - (R * T * p.ln_T_ref) - (R * T)));
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
        // c = sum_i rho_i, then the hoisted pre-calculations.
        Number c{0};
        for (std::size_t i = 0; i < this->size(); ++i) {
            c += rho_i[i];
        }
        const Number TlnT = T * std::log(T);
        const Number RTlnC = R * T * std::log(c);

        Number psi{0};
        for (std::size_t i = 0; i < this->size(); ++i) {
            const Params& p = params_[i];
            psi += (Number{2} * R * T * xlnx(rho_i[i])) +
                   (rho_i[i] * (p.h_ref + (p.c_p * (T - p.T_ref)) - (T * p.s_ref) + (p.c_p * T * p.ln_T_ref) - RTlnC -
                                (T * p.R_ln_c_ref) + ((R - p.c_p) * TlnT) - (R * T * p.ln_T_ref) - (R * T)));
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
        Number c{0};
        for (std::size_t i = 0; i < this->size(); ++i) {
            c += rho_i[i];
        }
        const Number TlnT = T * std::log(T);
        const Number RTlnC = R * T * std::log(c);

        for (std::size_t i = 0; i < this->size(); ++i) {
            const Params& p = params_[i];
            out[i] = (Number{2} * R * T * xlnx(rho_i[i])) +
                     (rho_i[i] * (p.h_ref + (p.c_p * (T - p.T_ref)) - (T * p.s_ref) + (p.c_p * T * p.ln_T_ref) - RTlnC -
                                  (T * p.R_ln_c_ref) + ((R - p.c_p) * TlnT) - (R * T * p.ln_T_ref) - (R * T)));
        }
    }

private:
    /**
     * @brief Stored (pre-computed) per-species parameters the kernels consume.
     *
     * Derived once, at construction, from the natural SpeciesInput.
     */
    struct Params {
        double T_ref;      ///< Reference temperature @f$T_\mathrm{ref}@f$ [K].
        double R_ln_c_ref; ///< @f$R\ln c_\mathrm{ref}@f$ [J/(mol K)].
        double c_p;        ///< Isobaric molar heat capacity @f$c_p@f$ [J/(mol K)].
        double ln_T_ref;   ///< @f$\ln T_\mathrm{ref}@f$ [-].
        double h_ref;      ///< Reference molar enthalpy @f$h_\mathrm{ref}@f$ [J/mol].
        double s_ref;      ///< Reference molar entropy @f$s_\mathrm{ref}@f$ [J/(mol K)].
    };

    /// @brief Array-of-Structs parameter storage: one @ref Params per species.
    using Storage = std::conditional_t<N == std::dynamic_extent, std::vector<Params>, std::array<Params, N>>;

    /// @brief Convert one set of natural inputs into the stored parameters.
    [[nodiscard]] static Params to_params(const SpeciesInput& in)
    {
        constexpr double R = ideal_gas_constant<double>;
        const double c_ref = in.p_ref / (R * in.T_ref);
        return Params{
            .T_ref = in.T_ref,
            .R_ln_c_ref = R * std::log(c_ref),
            .c_p = in.c_p,
            .ln_T_ref = std::log(in.T_ref),
            .h_ref = in.h_ref,
            .s_ref = in.s_ref,
        };
    }

    Storage params_{}; ///< Per-species stored parameters (AoS).
};

static_assert(IdealEoS<ConstantCp<2>>, "ConstantCp must satisfy the IdealEoS concept.");
static_assert(IdealEoS<ConstantCp<std::dynamic_extent>>, "ConstantCp must satisfy the IdealEoS concept.");
} // namespace glis::eos
