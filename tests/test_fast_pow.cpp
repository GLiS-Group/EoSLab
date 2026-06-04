#include "eoslab/core_calculations.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <limits>
#include <tuple>

using namespace boost::ut;

// Exercises the compile-time integer power helper glis::eos::detail::fast_pow.
// The previous implementation had a dead `N % 2 == 2` branch; these checks pin
// down correct results for even, odd, zero, one, and negative exponents.
int main()
{
    using glis::eos::detail::fast_pow;

    suite<"fast_pow"> s = [] {
        "matches std::pow for several exponents"_test = []<typename Number> {
            const Number base = Number{1.5};
            // Tolerance scaled to the type's precision: fast_pow uses repeated
            // multiplication while std::pow uses exp/log, so they differ by a
            // few ulps (which is ~1e-7 for float, ~1e-16 for double).
            const Number tol = Number{128} * std::numeric_limits<Number>::epsilon();
            const auto close = [tol](Number a, Number b) {
                return std::abs(a - b) <= tol * std::max(Number{1}, std::abs(b));
            };
            expect(close(fast_pow<Number, 0>(base), Number{1}));
            expect(close(fast_pow<Number, 1>(base), base));
            expect(close(fast_pow<Number, 2>(base), std::pow(base, Number{2})));
            expect(close(fast_pow<Number, 3>(base), std::pow(base, Number{3})));
            expect(close(fast_pow<Number, 4>(base), std::pow(base, Number{4})));
            expect(close(fast_pow<Number, 5>(base), std::pow(base, Number{5})));
            expect(close(fast_pow<Number, 8>(base), std::pow(base, Number{8})));
            expect(close(fast_pow<Number, -1>(base), Number{1} / base));
            expect(close(fast_pow<Number, -2>(base), Number{1} / (base * base)));
            expect(close(fast_pow<Number, -3>(base), std::pow(base, Number{-3})));
        } | std::tuple<float, double, long double>{};

        "even exponents are exact for integer bases"_test = [] {
            expect(fast_pow<double, 2>(3.0) == 9.0_d);
            expect(fast_pow<double, 4>(2.0) == 16.0_d);
            expect(fast_pow<double, 6>(2.0) == 64.0_d);
        };

        // fast_pow must be usable in a constant expression.
        "usable in constexpr"_test = [] {
            constexpr auto v = fast_pow<double, 4>(2.0);
            static_assert(v == 16.0);
            expect(v == 16.0_d);
        };
    };
}
