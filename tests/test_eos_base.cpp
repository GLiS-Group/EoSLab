//
// Tests for glis::eos::BaseEoS::for_each_component.
//
// for_each_component(f) must invoke f(idx) for idx = 0 .. size()-1 in ascending
// order, exactly once per component. Two regimes are exercised:
//   * compile-time size (N is a concrete value; the body is an unrolled fold),
//   * runtime size       (N == std::dynamic_extent; a runtime loop).
//
#include "eoslab/core/eos_base.hpp"

#include <boost/ut.hpp>
#include <cstddef>
#include <span>
#include <vector>

using namespace boost::ut;

int main()
{
    suite<"eos_base"> s = [] {
        // -------------------------------------------------------------------
        // Compile-time extent: every index visited once, in ascending order.
        // -------------------------------------------------------------------
        "compile-time extent visits each index once in order"_test = [] {
            constexpr std::size_t N = 5;
            const glis::eos::BaseEoS<N> base{};

            std::vector<std::size_t> visited;
            base.for_each_component([&](std::size_t i) { visited.push_back(i); });

            expect(eq(visited.size(), N));
            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(visited[i], i));
            }
        };

        // -------------------------------------------------------------------
        // Runtime extent: same contract, count taken from the constructor.
        // -------------------------------------------------------------------
        "dynamic extent visits each index once in order"_test = [] {
            const std::size_t n = 4;
            const glis::eos::BaseEoS<std::dynamic_extent> base{n};

            std::vector<std::size_t> visited;
            base.for_each_component([&](std::size_t i) { visited.push_back(i); });

            expect(eq(visited.size(), n));
            for (std::size_t i = 0; i < n; ++i) {
                expect(eq(visited[i], i));
            }
        };

        // -------------------------------------------------------------------
        // Single component (compile-time): exactly one call with index 0.
        // -------------------------------------------------------------------
        "single component visits index 0 exactly once"_test = [] {
            const glis::eos::BaseEoS<1> base{};

            std::size_t count = 0;
            std::size_t last = 999;
            base.for_each_component([&](std::size_t i) {
                ++count;
                last = i;
            });

            expect(eq(count, std::size_t{1}));
            expect(eq(last, std::size_t{0}));
        };

        // -------------------------------------------------------------------
        // Empty compile-time extent: the callable is never invoked.
        // -------------------------------------------------------------------
        "zero components never invokes the callable"_test = [] {
            const glis::eos::BaseEoS<0> base{};

            std::size_t count = 0;
            base.for_each_component([&](std::size_t /*i*/) { ++count; });

            expect(eq(count, std::size_t{0}));
        };
    };
}
