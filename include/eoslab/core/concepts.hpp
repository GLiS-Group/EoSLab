#pragma once
/**
 * @file concepts.hpp
 * @brief Concepts that define the interface every equation-of-state model must
 *        provide, and the ideal/residual classification.
 */
#include "eoslab/core/eos_base.hpp"

#include <concepts>
#include <cstddef>
namespace glis::eos {

/**
 * @def GLIS_EOS_RESTRICT
 * @brief Portable spelling of the C99 `restrict` pointer qualifier.
 *
 * Expands to the compiler's restrict keyword (`__restrict__` on GCC/Clang,
 * `__restrict` on MSVC) or to nothing on unknown compilers. Used to promise that
 * the buffers passed to the hot autodiff routines do not alias.
 */
#if defined(__GNUC__) || defined(__clang__)
#define GLIS_EOS_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define GLIS_EOS_RESTRICT __restrict
#else
#define GLIS_EOS_RESTRICT
#endif

/**
 * @brief Interface required of every equation-of-state model (ideal or residual).
 *
 * A conforming type @p E must expose the following const member functions. The
 * concept is checked with `double`, but models are expected to template these on
 * the floating-point type so the autodiff layer can also instantiate them with,
 * e.g., a higher-precision type.
 *
 * Required members:
 * - `calc_helmholtz(c, x, T) -> (convertible to double)`
 *   Molar Helmholtz energy @f$a@f$.
 *   - `c` molar concentration [mol/m^3]
 *   - `x` pointer to the mole-fraction array [-]
 *   - `T` temperature [K]
 *   - returns the **molar** Helmholtz energy [J/mol]
 * - `calc_partial_helmholtz(rho_i, T, out) -> void`
 *   Per-component decomposition of the Helmholtz energy **density** @f$\Psi@f$
 *   (such that @f$\sum_i \text{out}[i] = \Psi@f$).
 *   - `rho_i` pointer to the partial-molar-concentration array [mol/m^3]
 *   - `T` temperature [K]
 *   - `out` output array, per-component Helmholtz energy density [J/m^3]
 * - `size() -> (convertible to std::size_t)`
 *   Number of components [-].
 *
 * @note The two Helmholtz members use different units on purpose:
 *       `calc_helmholtz` is an intensive **molar** quantity [J/mol], while
 *       `calc_partial_helmholtz` returns a **volumetric** density [J/m^3]. They
 *       are linked by @f$\Psi(\rho,T) = c\,a(c,x,T)@f$ with @f$c=\sum_i\rho_i@f$
 *       and @f$x_i=\rho_i/c@f$.
 */
template<class E>
concept EquationOfState =
    requires(const E& eos, const double mole_concentration, const double* mole_fractions,
             const double* partial_mole_concentrations, const double temperature, double* out_array) {
        { eos.calc_partial_helmholtz(partial_mole_concentrations, temperature, out_array) } -> std::same_as<void>;
        { eos.calc_helmholtz(mole_concentration, mole_fractions, temperature) } -> std::convertible_to<double>;
        { eos.size() } -> std::convertible_to<std::size_t>;
    };

/**
 * @brief An ideal-gas contribution: an EquationOfState that also derives from
 *        BaseIdealEoS.
 *
 * The inheritance requirement is how the library tells ideal models apart from
 * residual ones at compile time.
 */
template<class E>
concept IdealEoS = std::derived_from<E, BaseIdealEoS> && EquationOfState<E>;

/**
 * @brief A residual (departure) contribution: an EquationOfState that is *not*
 *        an IdealEoS.
 *
 * Equivalently, a model that does not derive from BaseIdealEoS.
 */
template<class E>
concept ResidualEoS = EquationOfState<E> && !IdealEoS<E>;

} // namespace glis::eos