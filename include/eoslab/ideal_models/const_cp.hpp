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
 * ideal entropy of mixing) through glis::eos::MultiFluidBase.
 */

#include "eoslab/core/concepts.hpp"
#include "eoslab/core/multifluid_base.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/core/xlnx.hpp"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace glis::eos {

namespace detail {
/**
 * @brief Stored per-species parameters of the constant-@f$c_p@f$ model.
 *
 * These are the pre-computed quantities the kernels actually consume; they are
 * derived once, at construction, from the natural inputs in
 * glis::eos::ConstantCp::SpeciesInput. Must remain an all-`double` aggregate so
 * glis::eos::ParameterStorage can reflect over it.
 */
struct ConstantCpParams {
    double T_ref;      ///< Reference temperature @f$T_\mathrm{ref}@f$ [K].
    double R_ln_c_ref; ///< @f$R\ln c_\mathrm{ref}@f$ [J/(mol K)].
    double c_p;        ///< Isobaric molar heat capacity @f$c_p@f$ [J/(mol K)].
    double ln_T_ref;   ///< @f$\ln T_\mathrm{ref}@f$ [-].
    double h_ref;      ///< Reference molar enthalpy @f$h_\mathrm{ref}@f$ [J/mol].
    double s_ref;      ///< Reference molar entropy @f$s_\mathrm{ref}@f$ [J/(mol K)].
};
} // namespace detail

/**
 * @brief Ideal-gas equation of state with a constant isobaric molar heat
 *        capacity per species.
 *
 * Construct it from a per-species list of natural reference data
 * (SpeciesInput); the constructor performs the pre-calculations and forwards the
 * stored parameters to glis::eos::MultiFluidBase. Each species may use a
 * different reference temperature and pressure.
 *
 * @tparam N Component count, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent>
class ConstantCp : public MultiFluidBase<ConstantCp<N>, detail::ConstantCpParams, N>, public BaseIdealEoS {
    using Base = MultiFluidBase<ConstantCp<N>, detail::ConstantCpParams, N>;

public:
    /**
     * @brief Natural per-species reference data supplied by the user.
     *
     * The constructor turns these into the stored detail::ConstantCpParams. The
     * reference molar concentration is *not* supplied directly; it is computed as
     * @f$c_\mathrm{ref} = p_\mathrm{ref} / (R\,T_\mathrm{ref})@f$.
     */
    struct SpeciesInput {
        double T_ref; ///< Reference temperature @f$T_\mathrm{ref}@f$ [K].
        double p_ref; ///< Reference pressure @f$p_\mathrm{ref}@f$ [Pa].
        double c_p;   ///< Isobaric molar heat capacity @f$c_p@f$ [J/(mol K)].
        double h_ref; ///< Reference molar enthalpy @f$h_\mathrm{ref}@f$ [J/mol].
        double s_ref; ///< Reference molar entropy @f$s_\mathrm{ref}@f$ [J/(mol K)].
    };

    /// @brief Pre-calculation cached for the molar (c, x, T) kernel.
    template<std::floating_point Number> struct MolarPre {
        Number TlnT;  ///< @f$T\ln T@f$.
        Number RTlnC; ///< @f$R\,T\ln c@f$ (using the total molar concentration).
    };

    /// @brief Pre-calculation cached for the density (rho_i, T) kernel.
    template<std::floating_point Number> struct DensityPre {
        Number TlnT;  ///< @f$T\ln T@f$.
        Number RTlnC; ///< @f$R\,T\ln\!\left(\sum_i \rho_i\right)@f$.
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
        : Base(to_param_array(inputs))
    {
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
        : Base(std::span<const detail::ConstantCpParams>(to_param_vector(inputs)))
    {
    }

    /**
     * @brief Pre-calculation hoisted out of the molar component loop.
     * @param c Molar concentration [mol/m^3].
     * @param T Temperature [K].
     */
    template<std::floating_point Number>
    [[nodiscard]] [[gnu::always_inline]] MolarPre<Number> perform_pre_calculations(Number c, const Number* /*x*/,
                                                                                   Number T) const
    {
        return MolarPre<Number>{T * std::log(T), ideal_gas_constant<Number> * T * std::log(c)};
    }

    /**
     * @brief Pre-calculation hoisted out of the density component loop.
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     */
    template<std::floating_point Number>
    [[nodiscard]] [[gnu::always_inline]] DensityPre<Number> perform_pre_calculations(const Number* rho_i, Number T) const
    {
        Number c{0};
        this->for_each_component([&](std::size_t i) { c += rho_i[i]; });
        return DensityPre<Number>{T * std::log(T), ideal_gas_constant<Number> * T * std::log(c)};
    }

    /**
     * @brief Species @p i 's contribution to the molar Helmholtz energy.
     * @param x   Mole-fraction array [-].
     * @param T   Temperature [K].
     * @param i   Species index.
     * @param p   Stored parameters of species @p i.
     * @param pre Cached molar pre-calculation.
     */
    template<std::floating_point Number>
    [[nodiscard]] [[gnu::always_inline]] Number calc_helmholtz_i(Number /*c*/, const Number* x, Number T, std::size_t i,
                                                                 const detail::ConstantCpParams& p,
                                                                 const MolarPre<Number>& pre) const
    {
        const Number R = ideal_gas_constant<Number>;
        return (2 * R * T * xlnx(x[i]))
            + (x[i]
               * (p.h_ref + (p.c_p * (T - p.T_ref)) - (T * p.s_ref) + (p.c_p * T * p.ln_T_ref) + pre.RTlnC
                  - (T * p.R_ln_c_ref) + ((R - p.c_p) * pre.TlnT) - (R * T * p.ln_T_ref) - (R * T)));
    }

    /**
     * @brief Species @p i 's contribution to the Helmholtz energy density.
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @param i     Species index.
     * @param p     Stored parameters of species @p i.
     * @param pre   Cached density pre-calculation.
     */
    template<std::floating_point Number>
    [[nodiscard]] [[gnu::always_inline]] Number calc_helmholtz_density_i(const Number* rho_i, Number T, std::size_t i,
                                                                         const detail::ConstantCpParams& p,
                                                                         const DensityPre<Number>& pre) const
    {
        const Number R = ideal_gas_constant<Number>;
        return (2 * R * T * xlnx(rho_i[i]))
            + (rho_i[i]
               * (p.h_ref + (p.c_p * (T - p.T_ref)) - (T * p.s_ref) + (p.c_p * T * p.ln_T_ref) - pre.RTlnC
                  - (T * p.R_ln_c_ref) + ((R - p.c_p) * pre.TlnT) - (R * T * p.ln_T_ref) - (R * T)));
    }

private:
    /// @brief Convert one set of natural inputs into the stored parameters.
    [[nodiscard]] static detail::ConstantCpParams to_params(const SpeciesInput& in)
    {
        constexpr double R = ideal_gas_constant<double>;
        const double c_ref = in.p_ref / (R * in.T_ref);
        return detail::ConstantCpParams{
            .T_ref = in.T_ref,
            .R_ln_c_ref = R * std::log(c_ref),
            .c_p = in.c_p,
            .ln_T_ref = std::log(in.T_ref),
            .h_ref = in.h_ref,
            .s_ref = in.s_ref,
        };
    }

    /// @brief Transform a fixed-size input list into the stored parameter array.
    [[nodiscard]] static std::array<detail::ConstantCpParams, N>
    to_param_array(const std::array<SpeciesInput, N>& inputs)
    {
        std::array<detail::ConstantCpParams, N> out{};
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = to_params(inputs[i]);
        }
        return out;
    }

    /// @brief Transform a runtime-sized input list into the stored parameters.
    [[nodiscard]] static std::vector<detail::ConstantCpParams> to_param_vector(std::span<const SpeciesInput> inputs)
    {
        std::vector<detail::ConstantCpParams> out;
        out.reserve(inputs.size());
        for (const SpeciesInput& in : inputs) {
            out.push_back(to_params(in));
        }
        return out;
    }
};

static_assert(IdealEoS<ConstantCp<2>>, "ConstantCp must satisfy the IdealEoS concept.");
static_assert(IdealEoS<ConstantCp<std::dynamic_extent>>, "ConstantCp must satisfy the IdealEoS concept.");
} // namespace glis::eos
