#include "eoslab/core/horner.hpp"

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <cstddef>
#include <random>

namespace {
using namespace boost::ut;
template<int N, std::floating_point Number> void test_horner(const std::size_t n_tests = 10)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<Number> distrib(0, 10);
    std::uniform_real_distribution<Number> distrib2(0, 10);

    for (std::size_t i = 0; i < n_tests; ++i) {
        std::array<Number, N + 1> coeffs{};
        std::ranges::generate(coeffs, [&]() { return distrib(gen); });
        for (std::size_t j = 0; j < n_tests; ++j) {
            const Number x = distrib2(gen);
            Number y = coeffs[0];
            for (int k = 1; k < N + 1; ++k) {
                y += coeffs[k] * std::pow(x, k);
            }
            Number yh = glis::eos::eval_polynomial<N>(coeffs, x);
            expect(approx(yh, y, 1.e-9));
        }
    }
}
} // namespace

int main()
{
    using namespace boost::ut;

    suite<"Horner polynomial evaluation"> s = [] {
        "Number type"_test = []<typename T> {
            "Degree 0"_test = [] { test_horner<0, T>(); };
            "Degree 1"_test = [] { test_horner<1, T>(); };
            "Degree 2"_test = [] { test_horner<2, T>(); };
            "Degree 3"_test = [] { test_horner<3, T>(); };
            "Degree 4"_test = [] { test_horner<4, T>(); };
            "Degree 5"_test = [] { test_horner<5, T>(); };
        } | std::tuple<double, long double>{};
    };
}