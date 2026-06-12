#pragma once
/**
 * @file peng_robinson.hpp
 * @brief Residual (departure) model for the Peng-Robinson equation of state.
 */

#include "eoslab/core/concepts.hpp"
#include "eoslab/core/numbers.hpp"
#include "eoslab/residual_models/cubic.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace glis::eos {

/**
 * @brief Residual Helmholtz contribution of the Peng-Robinson equation of state.
 *
 * A glis::eos::BaseCubic model with @f$\Delta_{1,2} = 1 \pm \sqrt{2}@f$ and the
 * pure-species parameters built from each species' critical point and acentric
 * factor:
 * @f[
 *   \eta_c = \left(1 + \sqrt[3]{4 - \sqrt 8} + \sqrt[3]{4 + \sqrt 8}\right)^{-1},
 *   \qquad
 *   \Omega_a = \frac{8 + 40\eta_c}{49 - 37\eta_c},
 *   \qquad
 *   \Omega_b = \frac{\eta_c}{3 + \eta_c},
 * @f]
 * @f[
 *   a_{0,ii} = \Omega_a \frac{(R T_c)^2}{P_c},
 *   \qquad
 *   b_{ii} = \Omega_b \frac{R T_c}{P_c},
 * @f]
 * with the alpha-function slope from the acentric factor @f$\omega@f$:
 * @f[
 *   m_{ii} =
 *   \begin{cases}
 *     0.37464 + 1.54226\,\omega - 0.26992\,\omega^2, & \omega \le 0.491, \\
 *     0.379642 + 1.48503\,\omega - 0.164423\,\omega^2 + 0.016666\,\omega^3, & \omega > 0.491.
 *   \end{cases}
 * @f]
 * The Helmholtz kernels and mixture rules live in BaseCubic; this class only
 * performs the parameter transformation at construction.
 *
 * Pair the model with an ideal contribution in a glis::eos::EoS and evaluate
 * properties through the free functions in core_calculations.hpp:
 *
 * @code{.cpp}
 * using namespace glis::eos;
 *
 * // CH4 and CO2 from critical data, with one interaction coefficient.
 * PengRobinson<2> residual(
 *     std::array{PengRobinson<2>::SpeciesInput{.T_c = 190.564, .P_c = 4.5992e6, .omega = 0.011},
 *                PengRobinson<2>::SpeciesInput{.T_c = 304.1282, .P_c = 7.3773e6, .omega = 0.22394}},
 *     std::array{0.0, 0.09,
 *                0.09, 0.0});
 * EoS eos{some_ideal_model, residual};
 *
 * const std::array<double, 2> x{0.4, 0.6}; // mole fractions
 * const double p = calc_pressure(eos, 500.0, std::span<const double, 2>{x}, 300.0);
 * @endcode
 *
 * Use the @c std::dynamic_extent default (e.g. `PengRobinson<>`) when the
 * number of species is only known at run time.
 *
 * @tparam N Component count, or @c std::dynamic_extent for a runtime size.
 */
template<std::size_t N = std::dynamic_extent> class PengRobinson : public BaseCubic<PengRobinson<N>, N> {
public:
    /// @brief Natural per-species input: critical point and acentric factor.
    struct SpeciesInput {
        double T_c;   ///< Critical temperature @f$T_c@f$ [K].
        double P_c;   ///< Critical pressure @f$P_c@f$ [Pa].
        double omega; ///< Acentric factor @f$\omega@f$ [-].
    };

    /// @brief Generalized-cubic constant @f$\Delta_1 = 1 + \sqrt 2@f$.
    static constexpr double delta1 = 1.0 + std::numbers::sqrt2;
    /// @brief Generalized-cubic constant @f$\Delta_2 = 1 - \sqrt 2@f$.
    static constexpr double delta2 = 1.0 - std::numbers::sqrt2;

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
    explicit PengRobinson(const std::array<SpeciesInput, N>& inputs, const std::array<double, N * N>& kij = {})
        requires(N != std::dynamic_extent)
        : BaseCubic<PengRobinson<N>, N>(to_pure_all(inputs), kij)
    {
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
    explicit PengRobinson(std::span<const SpeciesInput> inputs, std::span<const double> kij = {})
        requires(N == std::dynamic_extent)
        : BaseCubic<PengRobinson<N>, N>(to_pure_all(inputs), kij)
    {
    }

private:
    using Base = BaseCubic<PengRobinson<N>, N>;
    using Pure = typename Base::PureSpecies;

    // Map one species' critical data to the generalized cubic parameters.
    static Pure to_pure(const SpeciesInput& in)
    {
        constexpr double R = ideal_gas_constant<double>;
        const double s8 = std::sqrt(8.0);
        const double eta_c = 1.0 / (1.0 + std::cbrt(4.0 - s8) + std::cbrt(4.0 + s8));
        const double omega_a = (8.0 + (40.0 * eta_c)) / (49.0 - (37.0 * eta_c));
        const double omega_b = eta_c / (3.0 + eta_c);
        const double w = in.omega;
        const double m = w <= 0.491 ? 0.37464 + (1.54226 * w) - (0.26992 * w * w)
                                    : 0.379642 + (1.48503 * w) - (0.164423 * w * w) + (0.016666 * w * w * w);
        const double RTc = R * in.T_c;
        return {.a0 = omega_a * RTc * RTc / in.P_c, .b = omega_b * RTc / in.P_c, .m = m, .T_c = in.T_c};
    }

    static std::array<Pure, N> to_pure_all(const std::array<SpeciesInput, N>& inputs)
        requires(N != std::dynamic_extent)
    {
        std::array<Pure, N> out{};
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = to_pure(inputs[i]);
        }
        return out;
    }

    static std::vector<Pure> to_pure_all(std::span<const SpeciesInput> inputs)
        requires(N == std::dynamic_extent)
    {
        std::vector<Pure> out;
        out.reserve(inputs.size());
        for (const SpeciesInput& in : inputs) {
            out.push_back(to_pure(in));
        }
        return out;
    }
};

static_assert(ResidualEoS<PengRobinson<2>>, "PengRobinson must satisfy the ResidualEoS concept.");
static_assert(ResidualEoS<PengRobinson<std::dynamic_extent>>, "PengRobinson must satisfy the ResidualEoS concept.");

} // namespace glis::eos
