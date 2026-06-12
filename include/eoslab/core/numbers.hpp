#pragma once

/**
 * @file numbers.hpp
 * @brief Fundamental physical constants used throughout the library.
 */

#include <concepts>
namespace glis::eos {

/**
 * @brief The universal (molar) gas constant @f$R@f$.
 *
 * @tparam Number Floating-point type the constant is expressed in.
 *
 * Value: `8.31446261815324` (the full CODATA value).
 * Units: **J / (mol K)** (joule per mole per kelvin).
 */
template<std::floating_point Number = double> inline constexpr Number ideal_gas_constant{8.31446261815324};

} // namespace glis::eos