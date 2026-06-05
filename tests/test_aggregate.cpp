//
// Tests for the compile-time aggregate-reflection utilities:
//   glis::eos::agg::aggregate_arity<T>()  -> member count of an aggregate
//   glis::eos::agg::make_from_indexed<T>  -> T{ access(0), access(1), ... }
//
// Checks are written as runtime expectations (not static_assert) so the file
// compiles against the Phase-1 stubs and fails at runtime until implemented.
//
#include "eoslab/core/aggregate.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cstddef>

using namespace boost::ut;
namespace agg = glis::eos::detail;

namespace {

struct OneField {
    double x;
};
struct TwoField {
    double h;
    double g;
};
struct FiveField {
    double a;
    double b;
    double c;
    double d;
    double e;
};
struct Empty {};
struct Mixed {
    double a;
    int b;
    double c;
};

} // namespace

int main()
{
    suite<"aggregate"> s = [] {
        // ----------------------------------------------------------------- //
        // aggregate_arity
        // ----------------------------------------------------------------- //
        "arity of a one-field struct is 1"_test = [] { expect(eq(agg::aggregate_arity<OneField>(), std::size_t{1})); };

        "arity of a two-field struct is 2"_test = [] { expect(eq(agg::aggregate_arity<TwoField>(), std::size_t{2})); };

        "arity of a five-field struct is 5"_test = [] {
            expect(eq(agg::aggregate_arity<FiveField>(), std::size_t{5}));
        };

        "arity of an empty struct is 0"_test = [] { expect(eq(agg::aggregate_arity<Empty>(), std::size_t{0})); };

        "arity of a mixed-type struct is 3"_test = [] { expect(eq(agg::aggregate_arity<Mixed>(), std::size_t{3})); };

        "aggregate_arity_v matches the function form"_test = [] {
            expect(eq(agg::aggregate_arity_v<TwoField>, std::size_t{2}));
        };

        // ----------------------------------------------------------------- //
        // make_from_indexed
        // ----------------------------------------------------------------- //
        "make_from_indexed fills members in order"_test = [] {
            const TwoField r = agg::make_from_indexed<TwoField>([](std::size_t p) { return static_cast<double>(p); });
            expect(eq(r.h, 0.0));
            expect(eq(r.g, 1.0));
        };

        "make_from_indexed round-trips a SoA-style accessor"_test = [] {
            // Two columns, two species; reconstruct species i = 1.
            const std::array<double, 2> col0{10.0, 20.0};
            const std::array<double, 2> col1{30.0, 40.0};
            const std::size_t i = 1;
            const TwoField r =
                agg::make_from_indexed<TwoField>([&](std::size_t p) { return p == 0 ? col0[i] : col1[i]; });
            expect(eq(r.h, 20.0));
            expect(eq(r.g, 40.0));
        };

        "make_from_indexed on a five-field struct"_test = [] {
            const FiveField r =
                agg::make_from_indexed<FiveField>([](std::size_t p) { return static_cast<double>(p * p); });
            expect(eq(r.a, 0.0));
            expect(eq(r.b, 1.0));
            expect(eq(r.c, 4.0));
            expect(eq(r.d, 9.0));
            expect(eq(r.e, 16.0));
        };
    };
}
