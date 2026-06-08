#pragma once

#include "eoslab/core/core_calculations.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>
namespace glis::eos {
namespace detail {
/**
 * @internal
 * @brief Compile-time binomial coefficient @f$\binom{n}{k}@f$.
 *
 * Evaluated with the multiplicative formula using the symmetry
 * @f$\binom{n}{k} = \binom{n}{n-k}@f$ to keep the running product small.
 *
 * @param n The size of the set.
 * @param k The number of elements chosen.
 * @return @f$\binom{n}{k}@f$, or @f$0@f$ when @p k > @p n.
 */
consteval unsigned long long choose(unsigned int n, unsigned int k)
{
    if (k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }

    k = std::min(k, n - k);

    unsigned long long result = 1;
    for (unsigned int i = 1; i <= k; ++i) {
        result *= (n - k + i);
        result /= i;
    }
    return result;
}

/**
 * @internal
 * @brief Coefficients of the inner polynomial of the order-@p N smoothstep.
 *
 * The order-@p N smoothstep is
 * @f[
 *   S_N(x) = x^{N+1} \sum_{n=0}^{N} c_n\, x^{n},
 *   \qquad c_n = (-1)^n \binom{N+n}{n}\binom{2N+1}{N-n},
 * @f]
 * the unique degree-@f$(2N+1)@f$ polynomial with @f$S_N(0)=0@f$, @f$S_N(1)=1@f$
 * and vanishing first @p N derivatives at both endpoints. This returns the
 * coefficients @f$c_0,\dots,c_N@f$ of the inner sum (the leading @f$x^{N+1}@f$
 * factor is applied separately in detail::smooth_step). The coefficients
 * alternate in sign, so a signed integer type is required.
 *
 * @tparam N The smoothness order (the result is @p N + 1 coefficients).
 * @return The coefficients @f$c_0,\dots,c_N@f$, in ascending power order.
 */
template<int N> consteval std::array<long long, N + 1> get_smooth_step_coeffs()
{
    std::array<long long, N + 1> coeffs{};
    for (int n = 0; n <= N; ++n) {
        long long coeff = n % 2 == 0 ? 1 : -1;
        coeff *= static_cast<long long>(choose(N + n, n));
        coeff *= static_cast<long long>(choose((2 * N) + 1, N - n));
        coeffs[n] = coeff;
    }
    return coeffs;
}

/**
 * @internal
 * @brief Generalized smoothstep @f$S_N@f$ of order @p N.
 *
 * Evaluates @f$S_N(x) = x^{N+1} \sum_{n=0}^{N} c_n x^n@f$ (see
 * detail::get_smooth_step_coeffs) using Horner's method for the inner sum and
 * detail::fast_pow for the @f$x^{N+1}@f$ factor. On @f$[0, 1]@f$ it ramps
 * monotonically from @f$S_N(0) = 0@f$ to @f$S_N(1) = 1@f$ with its first @p N
 * derivatives vanishing at both endpoints, so it is the @f$C^N@f$ "S-curve" used
 * to smooth the otherwise non-differentiable kink of xlnx() at the origin.
 *
 * @tparam N      The smoothness order (number of vanishing endpoint
 *                derivatives). Must be in @f$[0, 6]@f$.
 * @tparam Number A floating-point type.
 * @param  x      The argument; the smoothstep behaviour is intended for
 *                @f$x \in [0, 1]@f$.
 * @return @f$S_N(x)@f$.
 */
template<int N, std::floating_point Number> constexpr Number smooth_step(const Number x)
{
    static_assert(N >= 0, "The smoothness order N must be non-negative.");
    static_assert(N < 7, "Using a large value for N is almost certainly unnecessary.");
    constexpr auto coeffs = get_smooth_step_coeffs<N>();
    // Horner's method evaluation of the inner polynomial sum_{n} c_n x^n.
    auto result = static_cast<Number>(coeffs[N]);
    for (int idx = N - 1; idx >= 0; --idx) {
        result *= x;
        result += static_cast<Number>(coeffs[idx]);
    }
    // Apply the overall x^{N+1} factor carried by the smoothstep.
    result *= fast_pow<Number, N + 1>(x);
    return result;
}

} // namespace detail
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

/**
 * @brief Computes @f$ x \ln(x) @f$ with a smooth (@f$C^{\text{Continuity}}@f$)
 *        extension at and below zero.
 *
 * Like xlnx(Number) this is the continuous extension @f$x\ln x@f$ for @f$x > 0@f$
 * and @f$0@f$ for @f$x \le 0@f$, but the @p Continuity template parameter controls
 * how smoothly the two pieces are joined:
 * - `Continuity == 0` is exactly xlnx(Number): merely continuous. The
 *   derivative still diverges (@f$\ln x + 1 \to -\infty@f$) as @f$x \to 0^+@f$.
 * - `Continuity >= 1` multiplies @f$x\ln x@f$ by a smoothstep
 *   detail::smooth_step "ramp" on the tiny interval @f$x \in (0, \varepsilon)@f$
 *   (where @f$\varepsilon@f$ is the machine epsilon), pulling the contribution to
 *   zero so that the function and its first @p Continuity derivatives are
 *   continuous everywhere, including across the @f$x = 0@f$ join. For
 *   @f$x \ge \varepsilon@f$ the value is the unmodified @f$x\ln x@f$, so ordinary
 *   arguments are unaffected.
 *
 * Smoothing the @f$x\ln x@f$ kink is useful where the term and its derivatives are
 * fed to automatic differentiation (e.g. the entropy of mixing), so a mole
 * fraction approaching zero does not produce a divergent derivative.
 *
 * @tparam Continuity The order of continuity to enforce at the origin
 *                    (number of continuous derivatives). Must be non-negative;
 *                    see detail::smooth_step for the supported upper bound.
 * @tparam Number     A floating-point type.
 * @param x The argument.
 * @return The @f$C^{\text{Continuity}}@f$ extension of @f$x \ln(x)@f$.
 */
template<int Continuity, std::floating_point Number> Number xlnx(const Number x)
{
    static_assert(Continuity >= 0, "The continuity parameter must be non-negative!");
    if constexpr (Continuity == 0) {
        return xlnx(x);
    }
    else {
        constexpr auto eps = std::numeric_limits<Number>::epsilon();
        if (x <= Number{0}) {
            return Number{0};
        }
        if (x >= eps) {
            return x * std::log(x);
        }
        return detail::smooth_step<Continuity, Number>(x / eps) * x * std::log(x);
    }
}
} // namespace glis::eos