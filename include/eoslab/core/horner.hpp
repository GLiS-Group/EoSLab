#pragma once
/**
 * @file horner.hpp
 * @brief Compile-time-sized polynomial evaluation.
 */

#include "eoslab/core/attributes.hpp"

#include <array>
#include <concepts>

namespace glis::eos {
/**
 * @brief Evaluate a degree-@p N polynomial at @p x.
 *
 * Evaluates @f$\sum_{k=0}^{N} \text{coeffs}[k]\,x^k@f$ using Horner's method.
 *
 * @tparam N      Degree of the polynomial (`coeffs` holds @p N + 1 values).
 * @tparam Number A floating-point type.
 * @param coeffs Coefficients in ascending power order (`coeffs[k]` multiplies @f$x^k@f$).
 * @param x      The evaluation point.
 * @return The polynomial value.
 *
 * @code{.cpp}
 * std::array<double, 3> coeffs{1.0, 2.0, 3.0}; // 1 + 2x + 3x^2
 * double y = glis::eos::eval_polynomial<2>(coeffs, 2.0); // 17.0
 * @endcode
 */
template<int N, std::floating_point Number>
[[nodiscard]] GLIS_EOS_ALWAYS_INLINE constexpr Number eval_polynomial(std::array<Number, N + 1>& coeffs, Number x)
{
    Number p = coeffs[N];
    for (int idx = N - 1; idx >= 0; --idx) {
        p *= x;
        p += coeffs[idx];
    }
    return p;
}
} // namespace glis::eos