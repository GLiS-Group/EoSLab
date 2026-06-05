//
// Tests for glis::eos::ParameterStorage: SoA (column-major) per-species
// parameter storage built from a row list of named structs, exposing a named
// row view (get_parameters) and contiguous column views (column).
//
#include "eoslab/core/parameter_storage.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cstddef>
#include <span>
#include <vector>

using namespace boost::ut;
using glis::eos::ParameterStorage;

namespace {

struct Params {
    double h;
    double g;
};
struct Single {
    double x;
};
struct Triple {
    double a;
    double b;
    double c;
};

} // namespace

int main()
{
    suite<"parameter_storage"> s = [] {
        // ----------------------------------------------------------------- //
        // Fixed N: round-trip rows and columns.
        // ----------------------------------------------------------------- //
        "fixed-N get_parameters round-trips each row"_test = [] {
            const std::array<Params, 2> rows{Params{2.5, 1.5}, Params{3.1, 2.0}};
            const ParameterStorage<Params, 2> store{rows};

            const Params p0 = store.get_parameters(0);
            const Params p1 = store.get_parameters(1);
            expect(eq(p0.h, 2.5));
            expect(eq(p0.g, 1.5));
            expect(eq(p1.h, 3.1));
            expect(eq(p1.g, 2.0));
        };

        "fixed-N column returns each parameter across species"_test = [] {
            const std::array<Params, 2> rows{Params{2.5, 1.5}, Params{3.1, 2.0}};
            const ParameterStorage<Params, 2> store{rows};

            const std::span<const double, 2> c0 = store.column(0);
            const std::span<const double, 2> c1 = store.column(1);
            expect(eq(c0[0], 2.5));
            expect(eq(c0[1], 3.1));
            expect(eq(c1[0], 1.5));
            expect(eq(c1[1], 2.0));
        };

        "column memory is contiguous (SoA)"_test = [] {
            const std::array<Params, 2> rows{Params{2.5, 1.5}, Params{3.1, 2.0}};
            const ParameterStorage<Params, 2> store{rows};

            const std::span<const double, 2> c0 = store.column(0);
            expect(eq(&c0[1] - &c0[0], std::ptrdiff_t{1})); // adjacent species within a column
        };

        "fixed-N size reports N"_test = [] {
            const ParameterStorage<Params, 2> store{std::array<Params, 2>{Params{1.0, 2.0}, Params{3.0, 4.0}}};
            expect(eq(store.size(), std::size_t{2}));
        };

        // ----------------------------------------------------------------- //
        // Dynamic N: same contract, count from the row list.
        // ----------------------------------------------------------------- //
        "dynamic-N round-trips rows and columns"_test = [] {
            const std::vector<Params> rows{Params{2.5, 1.5}, Params{3.1, 2.0}, Params{4.2, 0.5}};
            const ParameterStorage<Params> store{std::span<const Params>{rows}};

            expect(eq(store.size(), std::size_t{3}));
            for (std::size_t i = 0; i < 3; ++i) {
                const Params p = store.get_parameters(i);
                expect(eq(p.h, rows[i].h));
                expect(eq(p.g, rows[i].g));
            }
            const std::span<const double> c0 = store.column(0);
            const std::span<const double> c1 = store.column(1);
            expect(eq(c0.size(), std::size_t{3}));
            expect(eq(c0[0], 2.5));
            expect(eq(c0[1], 3.1));
            expect(eq(c0[2], 4.2));
            expect(eq(c1[0], 1.5));
            expect(eq(c1[1], 2.0));
            expect(eq(c1[2], 0.5));
        };

        // ----------------------------------------------------------------- //
        // Arity wiring: 1- and 3-parameter structs.
        // ----------------------------------------------------------------- //
        "one-parameter struct"_test = [] {
            const ParameterStorage<Single, 2> store{std::array<Single, 2>{Single{10.0}, Single{20.0}}};
            expect(eq(store.parameter_count, std::size_t{1}));
            expect(eq(store.get_parameters(1).x, 20.0));
            expect(eq(store.column(0)[0], 10.0));
            expect(eq(store.column(0)[1], 20.0));
        };

        "three-parameter struct"_test = [] {
            const ParameterStorage<Triple, 2> store{
                std::array<Triple, 2>{Triple{1.0, 2.0, 3.0}, Triple{4.0, 5.0, 6.0}}};
            expect(eq(store.parameter_count, std::size_t{3}));
            const Triple t1 = store.get_parameters(1);
            expect(eq(t1.a, 4.0));
            expect(eq(t1.b, 5.0));
            expect(eq(t1.c, 6.0));
            expect(eq(store.column(2)[0], 3.0));
            expect(eq(store.column(2)[1], 6.0));
        };

        // ----------------------------------------------------------------- //
        // Single species (N = 1).
        // ----------------------------------------------------------------- //
        "single species round-trips"_test = [] {
            const ParameterStorage<Params, 1> store{std::array<Params, 1>{Params{7.0, 8.0}}};
            const Params p = store.get_parameters(0);
            expect(eq(p.h, 7.0));
            expect(eq(p.g, 8.0));
            expect(eq(store.column(0).size(), std::size_t{1}));
            expect(eq(store.column(1)[0], 8.0));
        };

        // ----------------------------------------------------------------- //
        // Inherited for_each_component visits every species once.
        // ----------------------------------------------------------------- //
        "for_each_component visits each species index once"_test = [] {
            const ParameterStorage<Params, 3> store{
                std::array<Params, 3>{Params{1, 1}, Params{2, 2}, Params{3, 3}}};
            std::vector<std::size_t> visited;
            store.for_each_component([&](std::size_t i) { visited.push_back(i); });
            expect(eq(visited.size(), std::size_t{3}));
            expect(eq(visited[0], std::size_t{0}));
            expect(eq(visited[1], std::size_t{1}));
            expect(eq(visited[2], std::size_t{2}));
        };
    };
}
