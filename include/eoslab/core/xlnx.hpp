#pragma once

#include <cmath>
#include <concepts>
namespace glis::eos {
/**
 * @brief Computes @f$ x \ln(x) @f$ with a continuous extension at and below zero.
 *
 * For @f$ x > 0 @f$ this returns @f$ x \ln(x) @f$. For @f$ x \le 0 @f$ it returns
 * @f$ 0 @f$, which provides the continuous extension @f$ \lim_{x \to 0^+} x \ln(x) = 0 @f$
 * at the origin and avoids passing non-positive arguments to `std::log`.
 *
 * This term arises frequently in the entropy of mixing and related
 * thermodynamic expressions, where it must remain well-defined as a
 * mole fraction or concentration approaches zero.
 *
 * @tparam Number A floating-point type.
 * @param x The argument.
 * @return @f$ x \ln(x) @f$ for @f$ x > 0 @f$, otherwise @f$ 0 @f$.
 */
template<std::floating_point Number> Number xlnx(const Number x)
{
    return x <= Number{0} ? Number{0} : x * std::log(x);
}
} // namespace glis::eos