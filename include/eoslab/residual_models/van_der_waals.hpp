#pragma once
/**
 * @file van_der_waals.hpp
 * @brief Residual (departure) model for the van der Waals equation of state.
 */

#include "eoslab/core/concepts.hpp"
#include "eoslab/core/eos_base.hpp"
#include "eoslab/core/numbers.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace glis::eos {

/**
 * @brief Residual Helmholtz contribution of the van der Waals equation of state.
 *
 * The molar residual Helmholtz energy is
 * @f[
 *   a_r = -R T \ln(1 - b_m c) - a_m c,
 * @f]
 * the @f$\Delta_1 = \Delta_2 = 0@f$ specialization of the generalized cubic
 * form (where @f$\psi_2@f$ degenerates to @f$c@f$; see glis::eos::BaseCubic for
 * the general case). The mixture parameters follow the one-fluid rules
 * @f[
 *   a_m = \sum_i \sum_j x_i x_j\, (1 - k_{ij}) \sqrt{a_{0,ii}\, a_{0,jj}},
 *   \qquad
 *   b_m = \sum_i x_i b_{ii},
 * @f]
 * with the pure-species parameters built from the critical point:
 * @f[
 *   a_{0,ii} = \frac{27 (R T_c)^2}{64 P_c}, \qquad b_{ii} = \frac{R T_c}{8 P_c}.
 * @f]
 * Since the vdW attractive term is temperature independent (@f$m_{ii}=0@f$),
 * the entire pair matrix is precomputed at construction and evaluation reduces
 * to the double sum plus one logarithm.
 *
 * Only the symmetric part @f$(k_{ij}+k_{ji})/2@f$ of the binary-interaction
 * coefficients can affect @f$a_m@f$, so an asymmetric input matrix is
 * symmetrized internally without loss.
 *
 * Pair the model with an ideal contribution in a glis::eos::EoS and evaluate
 * properties through the free functions in core_calculations.hpp:
 *
 * @code{.cpp}
 * using namespace glis::eos;
 *
 * // N2 and CO2 from their critical points, with one interaction coefficient.
 * VanDerWaals<2> residual(
 *     std::array{VanDerWaals<2>::SpeciesInput{.T_c = 126.192, .P_c = 3.3958e6},
 *                VanDerWaals<2>::SpeciesInput{.T_c = 304.1282, .P_c = 7.3773e6}},
 *     std::array{0.0, 0.05,
 *                0.05, 0.0});
 * EoS eos{some_ideal_model, residual};
 *
 * const std::array<double, 2> x{0.4, 0.6}; // mole fractions
 * const double p = calc_pressure(eos, 500.0, std::span<const double, 2>{x}, 300.0);
 * @endcode
 *
 * Use the @c std::dynamic_extent default (e.g. `VanDerWaals<>`) when the number
 * of species is only known at run time.
 *
 * @tparam N Component count, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent> class VanDerWaals : public BaseEoS<N> {
public:
    /// @brief Natural per-species input: the critical point.
    struct SpeciesInput {
        double T_c; ///< Critical temperature @f$T_c@f$ [K].
        double P_c; ///< Critical pressure @f$P_c@f$ [Pa].
    };

    /**
     * @brief Construct a compile-time-sized model from per-species critical data.
     *
     * Only available when the component count @p N is known at compile time.
     *
     * @param inputs One SpeciesInput per species.
     * @param kij    Full row-major @f$N \times N@f$ binary-interaction matrix
     *               @f$k_{ij}@f$ [-] (entry @c kij[i*N + j]); the diagonal must
     *               be zero. Defaults to all zeros. May be asymmetric; only the
     *               symmetric part affects the model.
     */
    explicit VanDerWaals(const std::array<SpeciesInput, N>& inputs, const std::array<double, N * N>& kij = {})
        requires(N != std::dynamic_extent)
    {
        init(inputs, kij);
    }

    /**
     * @brief Construct a runtime-sized model from per-species critical data.
     *
     * Only available when @p N is @c std::dynamic_extent; `size()` becomes
     * `inputs.size()`.
     *
     * @param inputs One SpeciesInput per species.
     * @param kij    Full row-major @f$n \times n@f$ binary-interaction matrix
     *               @f$k_{ij}@f$ [-], or an empty span for all zeros (the
     *               default). Size checked via @c assert.
     */
    explicit VanDerWaals(std::span<const SpeciesInput> inputs, std::span<const double> kij = {})
        requires(N == std::dynamic_extent)
        : BaseEoS<N>(inputs.size())
    {
        b_.resize(inputs.size());
        a_.resize(inputs.size() * inputs.size());
        init(inputs, kij);
    }

    /**
     * @brief Molar residual Helmholtz energy @f$a_r = -RT\ln(1-b_m c) - a_m c@f$.
     * @param c Molar concentration [mol/m^3]. Must satisfy @f$b_m c < 1@f$.
     * @param x Mole-fraction array [-].
     * @param T Temperature [K].
     * @return Molar residual Helmholtz energy [J/mol].
     */
    template<std::floating_point Number> [[nodiscard]] Number calc_helmholtz(Number c, const Number* x, Number T) const
    {
        const Number R = ideal_gas_constant<Number>;
        const std::size_t n = this->size();
        Number am{0};
        Number bm{0};
        for (std::size_t i = 0; i < n; ++i) {
            bm += x[i] * b_[i];
            Number row{0};
            for (std::size_t j = 0; j < n; ++j) {
                row += x[j] * a_[(i * n) + j];
            }
            am += x[i] * row;
        }
        // a_r = R T psi_1 - a_m psi_2 with psi_1 = -ln(1 - b_m c), psi_2 = c (vdW).
        return (-R * T * std::log(Number{1} - (bm * c))) - (am * c);
    }

    /**
     * @brief Total residual Helmholtz energy density @f$\Psi = c\,a_r@f$.
     *
     * Evaluated directly in partial concentrations as
     * @f$\Psi = -RTc\ln(1 - \sum_i \rho_i b_{ii}) - \sum_{ij}\rho_i\rho_j a_{ij}@f$,
     * which avoids forming mole fractions.
     *
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @return Residual Helmholtz energy density [J/m^3].
     */
    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density(const Number* rho_i, Number T) const
    {
        const Number R = ideal_gas_constant<Number>;
        const std::size_t n = this->size();
        Number c{0};
        Number bc{0}; // b_m * c = sum_i rho_i b_i
        Number ac{0}; // a_m * c^2 = sum_ij rho_i rho_j a_ij
        for (std::size_t i = 0; i < n; ++i) {
            c += rho_i[i];
            bc += rho_i[i] * b_[i];
            Number row{0};
            for (std::size_t j = 0; j < n; ++j) {
                row += rho_i[j] * a_[(i * n) + j];
            }
            ac += rho_i[i] * row;
        }
        return (-R * T * c * std::log(Number{1} - bc)) - ac;
    }

    /**
     * @brief Per-component residual Helmholtz energy density.
     *
     * The residual does not decompose naturally per component, so the
     * mole-fraction-weighted convention @f$\Psi_i = (\rho_i / c)\,\Psi@f$ is
     * used; it satisfies @f$\sum_i \Psi_i = \Psi@f$ by construction.
     *
     * @param rho_i Partial molar concentrations [mol/m^3].
     * @param T     Temperature [K].
     * @param[out] out Per-component Helmholtz energy density [J/m^3]; length `size()`.
     */
    template<std::floating_point Number> void calc_partial_helmholtz(const Number* rho_i, Number T, Number* out) const
    {
        const std::size_t n = this->size();
        Number c{0};
        for (std::size_t i = 0; i < n; ++i) {
            c += rho_i[i];
        }
        // Mole-fraction-weighted decomposition: out[i] = (rho_i / c) * Psi.
        const Number scale = calc_helmholtz_density(rho_i, T) / c;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = rho_i[i] * scale;
        }
    }

private:
    using Vec = std::conditional_t<N == std::dynamic_extent, std::vector<double>, std::array<double, N>>;
    using Mat = std::conditional_t<N == std::dynamic_extent, std::vector<double>, std::array<double, N * N>>;

    // Build the stored parameters from critical data. Cold path: the pairwise
    // square roots are paid once here so the kernels never take any.
    void init(std::span<const SpeciesInput> inputs, std::span<const double> kij)
    {
        constexpr double R = ideal_gas_constant<double>;
        const std::size_t n = inputs.size();
        assert(kij.empty() || kij.size() == n * n);

        for (std::size_t i = 0; i < n; ++i) {
            b_[i] = R * inputs[i].T_c / (8.0 * inputs[i].P_c);
        }
        const auto a0 = [&](std::size_t i) {
            const double RTc = R * inputs[i].T_c;
            return 27.0 * RTc * RTc / (64.0 * inputs[i].P_c);
        };
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                // Only the symmetric part of k_ij can affect a_m = sum_ij x_i x_j a_ij.
                const double k_sym = kij.empty() ? 0.0 : 0.5 * (kij[(i * n) + j] + kij[(j * n) + i]);
                a_[(i * n) + j] = (1.0 - k_sym) * std::sqrt(a0(i) * a0(j));
            }
        }
    }

    Vec b_{}; // b_ii covolumes [m^3/mol].
    Mat a_{}; // a_ij = (1 - (k_ij + k_ji)/2) * sqrt(a0_ii * a0_jj), row-major [J m^3/mol^2].
              // T-independent for vdW (m_ii = 0), so fully precomputed at construction.
};

static_assert(ResidualEoS<VanDerWaals<2>>, "VanDerWaals must satisfy the ResidualEoS concept.");
static_assert(ResidualEoS<VanDerWaals<std::dynamic_extent>>, "VanDerWaals must satisfy the ResidualEoS concept.");

} // namespace glis::eos
