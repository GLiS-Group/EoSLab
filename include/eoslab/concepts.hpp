#pragma once
#include "eoslab/eos_base.hpp"

#include <concepts>
#include <cstddef>
namespace glis::eos {

#if defined(__GNUC__) || defined(__clang__)
#define GLIS_EOS_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define GLIS_EOS_RESTRICT __restrict
#else
#define GLIS_EOS_RESTRICT
#endif

template<class E>
concept EquationOfState = requires(const E& eos, const double mole_concentration, const double* mole_fractions,
                                   const double* partial_mole_concentrations, const double temperature, double* out_array) {
    { eos.calc_partial_helmholtz(partial_mole_concentrations, temperature, out_array) } -> std::same_as<void>;
    { eos.calc_helmholtz(mole_concentration, mole_fractions, temperature) } -> std::convertible_to<double>;
    { eos.size() } -> std::convertible_to<std::size_t>;
};

template<class E>
concept IdealEoS = std::derived_from<E, BaseIdealEoS> && EquationOfState<E>;

template<class E>
concept ResidualEoS = EquationOfState<E> && !IdealEoS<E>;

} // namespace glis::eos