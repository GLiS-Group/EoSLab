#include "eoslab/core/xlnx.hpp"

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>

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
        } | std::tuple<float, double, long double>{};
    };
}
