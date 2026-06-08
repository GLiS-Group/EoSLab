#include "eoslab/core/xlnx.hpp"

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

namespace {
/// @brief Evaluate @f$x^k@f$ for a non-negative integer exponent.
template<typename Number> constexpr Number ipow(const Number x, const int k)
{
    Number result{1};
    for (int i = 0; i < k; ++i) {
        result *= x;
    }
    return result;
}

/// @brief Assert that `get_smooth_step_coeffs<N>()` matches the expected coefficients exactly.
template<int N> void expect_coeffs(const std::array<long long, N + 1>& expected)
{
    constexpr auto coeffs = glis::eos::detail::get_smooth_step_coeffs<N>();
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        boost::ut::expect(boost::ut::eq(coeffs[i], expected[i])) << "N = " << N << ", coefficient " << i;
    }
}
} // namespace

int main()
{
    using namespace boost::ut;
    using glis::eos::xlnx;

    suite<"xlnx"> s = [] {
        constexpr unsigned int n_tests = 100;
        std::random_device rd;
        std::mt19937 gen(rd());

        "Scalar type"_test = [&]<typename Number> {
            std::uniform_real_distribution<Number> positive(std::numeric_limits<Number>::min(),
                                                            std::numeric_limits<Number>::max());
            std::array<Number, n_tests> positive_x{};
            std::ranges::generate(positive_x, [&]() { return positive(gen); });
            "x > 0"_test = [&](const auto x) { expect(eq(xlnx(x), x * std::log(x))); } | positive_x;

            " x == 0"_test = [] {
                const Number x{0};
                expect(eq(xlnx(x), Number{0}));
            };

            std::uniform_real_distribution<Number> small(std::numeric_limits<Number>::min(), Number{1});
            std::array<Number, n_tests> small_x{};
            std::ranges::generate(small_x, [&]() { return small(gen); });
            " 0 < x < 1"_test = [&](const auto x) { expect(lt(xlnx(x), Number{0})); } | small_x;

            std::array<Number, n_tests> negative_x{};
            std::ranges::generate(negative_x, [&]() { return -positive(gen); });
            " x < 0 "_test = [](const auto x) { expect(eq(xlnx(x), Number{0})); } | negative_x;

            // --- Smooth (C^Continuity) variant ----------------------------------------

            // Continuity == 0 must reproduce the plain continuous extension bit-for-bit.
            "xlnx<0> == xlnx (x > 0)"_test = [&](const auto x) { expect(eq(xlnx<0>(x), xlnx(x))); } | positive_x;
            "xlnx<0> == xlnx (x <= 0)"_test = [] {
                expect(eq(xlnx<0>(Number{0}), xlnx(Number{0})));
                expect(eq(xlnx<0>(Number{-1}), xlnx(Number{-1})));
            };

            // Away from zero (x >= epsilon) the smooth variants are the unmodified x*ln(x).
            "smooth xlnx == x ln x away from zero"_test = [&](const auto x) {
                expect(eq(xlnx<1>(x), x * std::log(x)));
                expect(eq(xlnx<2>(x), x * std::log(x)));
                expect(eq(xlnx<3>(x), x * std::log(x)));
            } | positive_x;

            // At and below zero the smooth variants are still exactly zero.
            "smooth xlnx at and below zero"_test = [] {
                expect(eq(xlnx<1>(Number{0}), Number{0}));
                expect(eq(xlnx<2>(Number{0}), Number{0}));
                expect(eq(xlnx<3>(Number{-1}), Number{0}));
            };

            // The piecewise join sits at x == epsilon, where the smoothstep equals 1, so the
            // function value is continuous with the x*ln(x) branch.
            "smooth xlnx is continuous at the join"_test = [] {
                constexpr Number eps = std::numeric_limits<Number>::epsilon();
                expect(eq(xlnx<1>(eps), eps * std::log(eps)));
                expect(eq(xlnx<3>(eps), eps * std::log(eps)));
            };

            // Inside the smoothing interval (0, epsilon) the smoothstep ramp in [0, 1] only
            // shrinks the (negative) x*ln(x) contribution towards zero.
            "smooth xlnx is bounded in the smoothing interval"_test = [] {
                constexpr Number eps = std::numeric_limits<Number>::epsilon();
                const std::array<Number, 6> fractions{Number{0.01}, Number{0.1},  Number{0.25},
                                                      Number{0.5},  Number{0.75}, Number{0.9}};
                for (const Number fraction : fractions) {
                    const Number x = fraction * eps;
                    const Number plain = x * std::log(x); // negative since 0 < x < 1
                    const Number smooth = xlnx<1>(x);
                    expect(le(smooth, Number{0}));
                    expect(ge(smooth, plain));
                }
            };
        } | std::tuple<float, double, long double>{};

        // The order-N smoothstep coefficients (inner polynomial, ascending powers) must match
        // the known closed forms. With the x^{N+1} factor restored these are the first seven
        // smoothstep polynomials S_0..S_6.
        "smooth_step coefficients"_test = [] {
            expect_coeffs<0>({1});                                             // S_0 = x
            expect_coeffs<1>({3, -2});                                         // S_1 = 3x^2 - 2x^3
            expect_coeffs<2>({10, -15, 6});                                    // S_2 = 10x^3 - 15x^4 + 6x^5
            expect_coeffs<3>({35, -84, 70, -20});                              // S_3
            expect_coeffs<4>({126, -420, 540, -315, 70});                      // S_4
            expect_coeffs<5>({462, -1980, 3465, -3080, 1386, -252});           // S_5
            expect_coeffs<6>({1716, -9009, 20020, -24024, 16380, -6006, 924}); // S_6
        };

        // Evaluate smooth_step<N> and compare against the seven reference polynomials given in
        // full form. Restricted to double/long double because the high-degree polynomials suffer
        // heavy cancellation near x = 1 that swamps float precision.
        "smooth_step evaluation"_test = [&]<typename Number> {
            using glis::eos::detail::smooth_step;
            const Number tol = std::is_same_v<Number, double> ? Number{1e-9} : Number{1e-11};
            const std::array<Number, 7> xs{Number{0},    Number{0.1}, Number{0.25}, Number{0.5},
                                           Number{0.75}, Number{0.9}, Number{1}};
            for (const Number x : xs) {
                expect(lt(std::abs(smooth_step<0>(x) - (x)), tol));
                expect(lt(std::abs(smooth_step<1>(x) - (-2 * ipow(x, 3) + 3 * ipow(x, 2))), tol));
                expect(lt(std::abs(smooth_step<2>(x) - (6 * ipow(x, 5) - 15 * ipow(x, 4) + 10 * ipow(x, 3))), tol));
                expect(lt(std::abs(smooth_step<3>(x) -
                                   (-20 * ipow(x, 7) + 70 * ipow(x, 6) - 84 * ipow(x, 5) + 35 * ipow(x, 4))),
                          tol));
                expect(lt(std::abs(smooth_step<4>(x) - (70 * ipow(x, 9) - 315 * ipow(x, 8) + 540 * ipow(x, 7) -
                                                        420 * ipow(x, 6) + 126 * ipow(x, 5))),
                          tol));
                expect(lt(std::abs(smooth_step<5>(x) - (-252 * ipow(x, 11) + 1386 * ipow(x, 10) - 3080 * ipow(x, 9) +
                                                        3465 * ipow(x, 8) - 1980 * ipow(x, 7) + 462 * ipow(x, 6))),
                          tol));
                expect(lt(std::abs(smooth_step<6>(x) -
                                   (924 * ipow(x, 13) - 6006 * ipow(x, 12) + 16380 * ipow(x, 11) - 24024 * ipow(x, 10) +
                                    20020 * ipow(x, 9) - 9009 * ipow(x, 8) + 1716 * ipow(x, 7))),
                          tol));
            }

            // Endpoint properties of every smoothstep: S_N(0) == 0 exactly and S_N(1) == 1.
            expect(eq(smooth_step<3>(Number{0}), Number{0}));
            expect(eq(smooth_step<6>(Number{0}), Number{0}));
            expect(lt(std::abs(smooth_step<3>(Number{1}) - Number{1}), tol));
            expect(lt(std::abs(smooth_step<6>(Number{1}) - Number{1}), tol));
        } | std::tuple<double, long double>{};
    };
}
