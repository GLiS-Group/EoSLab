#include "eos_test_models.hpp"
#include "eoslab/concepts.hpp"

#include <boost/ut.hpp>

using namespace boost::ut;
using namespace eoslab_test;
namespace ge = glis::eos;

// A type that does not model EquationOfState at all.
namespace  {
struct NotAnEoS {
    static int foo()  { return 0; }
};
}

// Compile-time verification of the concept hierarchy. If any of these fire, the
// concepts in concepts.hpp (or the test models) regressed.
static_assert(ge::EquationOfState<IdealGasTestModel<2>>);
static_assert(ge::EquationOfState<VirialResidualTestModel<2>>);

static_assert(ge::IdealEoS<IdealGasTestModel<2>>);
static_assert(!ge::IdealEoS<VirialResidualTestModel<2>>); // not derived from BaseIdealEoS

static_assert(ge::ResidualEoS<VirialResidualTestModel<2>>);
static_assert(!ge::ResidualEoS<IdealGasTestModel<2>>); // it IS an ideal EoS

static_assert(!ge::EquationOfState<NotAnEoS>);
static_assert(!ge::IdealEoS<NotAnEoS>);
static_assert(!ge::ResidualEoS<NotAnEoS>);

// The dynamic-extent variants also model the concepts.
static_assert(ge::IdealEoS<IdealGasTestModel<3>>);
static_assert(ge::ResidualEoS<VirialResidualTestModel<3>>);


int main()
{
    suite<"concepts"> s = [] {
        "models satisfy the expected concepts"_test = [] {
            // The static_asserts above already enforce this at compile time;
            // mirror them at runtime so the test reports as executed.
            expect(ge::IdealEoS<IdealGasTestModel<2>>);
            expect(ge::ResidualEoS<VirialResidualTestModel<2>>);
            expect(not ge::EquationOfState<NotAnEoS>);
        };

        "EoS pair exposes size and members"_test = [] {
            auto eos = make_binary_model();
            expect(eos.size() == 2_ul);
        };
    };
}
