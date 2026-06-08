# Implementing a new equation of state

This guide explains how to add a new equation-of-state (EoS) model to EoSLab so
that it works with the thermodynamic property routines in
[`core_calculations.hpp`](../include/eoslab/core/core_calculations.hpp).

## 1. The big picture

A complete equation of state in this library is a **pair**: one *ideal*
contribution and one *residual* (departure) contribution, joined by
`glis::eos::EoS<Ideal, Residual>` (see
[`eos_pair.hpp`](../include/eoslab/core/eos_pair.hpp)).

$$
\text{total Helmholtz energy} = a_\text{total} = a_\text{ideal} + a_\text{residual}
$$

Every property (pressure, enthalpy, heat capacities, chemical potentials, …) is
computed from the reduced molar Helmholtz energy

$$
\alpha(c,\vec x, T) = \frac{a(c,\vec x,T)}{RT}
$$

and its derivatives, which are obtained automatically with
[Enzyme](https://enzyme.mit.edu). **You never write a derivative by hand** — you
only implement the Helmholtz energy, and the library differentiates it.

### Symbols and units

| Symbol  | Meaning                                   | Units        |
|---------|-------------------------------------------|--------------|
| `c`     | molar concentration (molar density)       | mol/m³       |
| `x`     | mole fractions                            | – (sum to 1) |
| `T`     | temperature                               | K            |
| `rho_i` | partial molar concentrations              | mol/m³       |
| `R`     | gas constant (`glis::eos::ideal_gas_constant`) | J/(mol·K) |

## 2. The interface every model must provide

A model must satisfy the `glis::eos::EquationOfState` concept
([`concepts.hpp`](../include/eoslab/core/concepts.hpp)), i.e. expose these **const**
members. Template them on the floating-point type — the library instantiates
them with `double` and the test-suite also uses `long double`.

```cpp
// Molar Helmholtz energy a(c, x, T).
//   c : molar concentration   [mol/m^3]
//   x : mole-fraction array    [-]
//   T : temperature            [K]
//   returns the MOLAR Helmholtz energy [J/mol]
template<std::floating_point Number>
Number calc_helmholtz(Number c, const Number* x, Number T) const;

// Total Helmholtz energy DENSITY Psi(rho, T), accumulated into one scalar.
// This is the scalar reverse-mode autodiff differentiates for chemical potentials.
//   rho_i : partial-molar-concentration array [mol/m^3]
//   T     : temperature                       [K]
//   returns the total Helmholtz energy density [J/m^3]
template<std::floating_point Number>
Number calc_helmholtz_density(const Number* rho_i, Number T) const;

// Per-component decomposition of the Helmholtz energy DENSITY Psi.
//   out : output array; out[i] = per-component Helmholtz density [J/m^3]
//         such that sum_i out[i] = Psi(rho, T)
template<std::floating_point Number>
void calc_partial_helmholtz(const Number* rho_i, Number T, Number* out) const;

// Number of components [-].
std::size_t size() const;   // (provided for you by BaseEoS<N>)
```

> **Mind the units.** `calc_helmholtz` returns an *intensive molar* energy
> [J/mol], whereas `calc_helmholtz_density` / `calc_partial_helmholtz` return a
> *volumetric density* [J/m³]. They must be consistent:
>
> $$
> \Psi(\vec\rho,T) = c\,a(c,\vec{x},T) \quad \text{with} \quad c = \sum_i \rho_i,\quad x_i=\frac{\rho_i}{c}
> $$
>
> Implement them from a single definition so this relationship holds exactly.

You rarely write all four members by hand. The **recommended** path is to derive
from one of the CRTP bases in §3, which implement them for you from a small
per-species kernel. Implementing the concept directly (§3c) is the fallback for
models that do not fit either base.

## 3. Pick an implementation strategy

| Your model is…                                                            | Use                  |
|---------------------------------------------------------------------------|----------------------|
| a sum of per-species contributions, `a = Σ_i a_i`                         | `MultiFluidBase` (§3a) |
| a pseudo-pure fluid: mix the parameters, then evaluate one bulk expression | `OneFluidBase` (§3b)   |
| neither                                                                   | the concept directly (§3c) |

Both CRTP bases inherit Structure-of-Arrays per-species parameter storage
([`parameter_storage.hpp`](../include/eoslab/core/parameter_storage.hpp)) and
provide `calc_helmholtz`, `calc_helmholtz_density`, `calc_partial_helmholtz`, and
`size()`. You supply only a parameter struct and the per-species math.

### 3a. A multi-fluid model (the common case)

See [`MultiFluidBase`](../include/eoslab/core/multifluid_base.hpp) and the worked
example [`const_cp.hpp`](../include/eoslab/ideal_models/const_cp.hpp).

1. Define a **parameter struct**: a flat, trivially-copyable, all-`double`
   aggregate holding one species' parameters. Its member count is discovered
   automatically by reflection.
2. Derive from `MultiFluidBase<Derived, ParamStruct, N>` (CRTP — pass your own
   type), plus `BaseIdealEoS` if the model is ideal (see §4).
3. Provide two **public** per-species kernels (public so the CRTP downcast can
   call them):
   - `calc_helmholtz_i(c, x, T, i, const ParamStruct&)` → species `i`'s
     contribution to the molar Helmholtz energy,
   - `calc_helmholtz_density_i(rho_i, T, i, const ParamStruct&)` → species `i`'s
     contribution to the Helmholtz density.
4. **Optionally** hoist a value computed once and reused for every species (e.g.
   `ln T`, or a mole-fraction / concentration sum): define **both**
   `perform_pre_calculations(c, x, T)` and `perform_pre_calculations(rho_i, T)`,
   each returning a `Number`-templated struct, and add a trailing
   `const Pre<Number>&` argument to the two kernels. The base detects these
   automatically; omit them (and the extra argument) if you don't need them.
5. **Mark the kernels and `perform_pre_calculations` `GLIS_EOS_ALWAYS_INLINE`**
   (see §5 — this is required for correct autodiff, not just performance).

```cpp
#include "eoslab/core/attributes.hpp"
#include "eoslab/core/multifluid_base.hpp"

struct MyParams { double a; double b; };   // one species' parameters

template<std::size_t N>
class MyIdeal : public glis::eos::MultiFluidBase<MyIdeal<N>, MyParams, N>,
                public glis::eos::BaseIdealEoS {
public:
    using Base = glis::eos::MultiFluidBase<MyIdeal<N>, MyParams, N>;
    using Base::Base;   // inherit the constructors (see "Construction" below)

    template<std::floating_point Number>
    [[nodiscard]] GLIS_EOS_ALWAYS_INLINE
    Number calc_helmholtz_i(Number c, const Number* x, Number T, std::size_t i,
                            const MyParams& p) const { /* ... */ }

    template<std::floating_point Number>
    [[nodiscard]] GLIS_EOS_ALWAYS_INLINE
    Number calc_helmholtz_density_i(const Number* rho_i, Number T, std::size_t i,
                                    const MyParams& p) const { /* ... */ }
};
```

**Construction.** `MultiFluidBase` re-exports the storage constructors via
`using Base::Base;`: build from `std::array<ParamStruct, N>` (compile-time `N`)
or `std::span<const ParamStruct>` (`std::dynamic_extent`). If your users think in
*natural* inputs that differ from what you want to store, give the model its own
constructor that converts them and forwards the stored struct to the base — see
`ConstantCp::SpeciesInput` and its `to_param_array` /`to_param_vector` helpers in
[`const_cp.hpp`](../include/eoslab/ideal_models/const_cp.hpp).

### 3b. A one-fluid model

See [`OneFluidBase`](../include/eoslab/core/onefluid_base.hpp). A one-fluid model
collapses the mixture to a pseudo-pure fluid: a **mixing rule** averages the
per-species parameters as functions of composition, then a single **bulk**
expression is evaluated with those averaged parameters. Derive from
`OneFluidBase<Derived, ParamStruct, N>` and provide (all public):

- the mixing rule `perform_pre_calculations(c, x, T)` and
  `perform_pre_calculations(rho_i, T)`, returning a `Number`-templated
  averaged-parameter struct (use `column(p)` for the contiguous reduction over
  species);
- the bulk kernels `calc_helmholtz_bulk(c, x, T, const Avg<Number>&)` and
  `calc_helmholtz_density_bulk(rho_i, T, const Avg<Number>&)`.

As in §3a, **mark the mixing rule and bulk kernels `GLIS_EOS_ALWAYS_INLINE`**.

### 3c. From scratch

If neither base fits, implement the four concept members from §2 directly and
derive from `glis::eos::BaseEoS<N>` for the component-count plumbing (`size()`,
`for_each_component`). Use a concrete `N` for a compile-time component count, or
`std::dynamic_extent` for a runtime one. See
[`no_residual.hpp`](../include/eoslab/residual_models/no_residual.hpp) and the two
worked models in [`tests/eos_test_models.hpp`](../tests/eos_test_models.hpp).
These are the differentiated entry points, so they do **not** need
`GLIS_EOS_ALWAYS_INLINE` themselves; but mark any helper they call that takes or
returns a parameter / pre-calculation aggregate (see §5).

## 4. Ideal vs. residual

### `IdealEoS`

```cpp
template<class E>
concept IdealEoS = std::derived_from<E, BaseIdealEoS> && EquationOfState<E>;
```

An ideal model must:

1. **Derive publicly from `glis::eos::BaseIdealEoS`** (the tag that marks it as
   ideal), and
2. satisfy `EquationOfState`.

Additionally, for the property routines to be physically consistent, a genuine
ideal gas must have compressibility factor `Z = 1`, which means its molar
Helmholtz energy depends on concentration as

$$
\left(\frac{\partial a_\text{ideal}}{\partial c}\right)_{T,\vec x} = \frac{RT}{c} \iff a_\text{ideal} = RT \ln c + f(T,\vec x)
$$

EoSLab makes this assumption internally to simplify calculations. The reference
for the derivative calculations we implemented may be found
[here](https://doi.org/10.1021/acs.iecr.2c00237).

### `ResidualEoS`

```cpp
template<class E>
concept ResidualEoS = EquationOfState<E> && !IdealEoS<E>;
```

A residual model must satisfy `EquationOfState` and **must not** derive from
`BaseIdealEoS` — that is the only thing distinguishing it from an ideal model at
compile time. Its `calc_helmholtz` returns the *residual* molar Helmholtz energy
(the departure from ideal-gas behaviour), with `alpha_res -> 0` as `c -> 0`.

## 5. `GLIS_EOS_ALWAYS_INLINE` and correct autodiff

This is **not** merely a performance hint. Enzyme differentiates LLVM IR and
relies on inlining + scalarization having already flattened the expression. It
cannot follow your model's per-species parameter struct or hoisted
pre-calculation struct when they cross **un-inlined call boundaries** as
aggregates: its type/activity analysis loses them, and at low optimization levels
(the `-O1` **debug** build) it silently produces **wrong derivatives** — mild
errors in forward mode, garbage in reverse mode — even though `-O3` happens to
inline them away and works.

The fix is to force every function *called from within* the differentiated entry
points to inline, so those aggregates never cross a call boundary and `mem2reg`
can promote them to registers before Enzyme runs. Mark with
`GLIS_EOS_ALWAYS_INLINE` (from
[`attributes.hpp`](../include/eoslab/core/attributes.hpp)):

- **multi-fluid:** your per-species kernels and `perform_pre_calculations`;
- **one-fluid:** your mixing rule and bulk kernels;
- **from scratch:** any helper your `calc_helmholtz` / `calc_helmholtz_density`
  call that takes or returns a parameter / pre-calculation aggregate (a model
  that inlines all of its math directly into those functions needs nothing extra).

The CRTP bases already mark their shared plumbing (`for_each_component`,
`get_parameters`). Note that the differentiated **entry points themselves** —
`calc_helmholtz` / `calc_helmholtz_density` — do *not* need the attribute; Enzyme
differentiates each as a standalone function, and only the intermediate calls
they make need to be flattened into them. The macro expands to
`[[clang::always_inline]]` on Clang and to nothing elsewhere.

> If you ever see derivative-based properties that pass in `release` but fail in
> `debug`, a missing `GLIS_EOS_ALWAYS_INLINE` on a kernel or pre-calculation is
> the first thing to check.

## 6. Add a `static_assert` (recommended)

Right below your model, assert that it satisfies the concept you intend — this
turns a subtle interface mistake (wrong signature, missing `const`, forgetting
to derive from `BaseIdealEoS`, …) into an immediate, readable compile error:

```cpp
#include "eoslab/core/concepts.hpp"

static_assert(glis::eos::IdealEoS<MyIdeal<2>>,
              "MyIdeal must satisfy the IdealEoS concept");
static_assert(glis::eos::IdealEoS<MyIdeal<std::dynamic_extent>>,
              "MyIdeal must satisfy the IdealEoS concept");
```

(Use `ResidualEoS` for a residual model.) See
[`tests/eos_test_models.hpp`](../tests/eos_test_models.hpp) for worked examples
and [`tests/test_concepts.cpp`](../tests/test_concepts.cpp) for concept assertions.

## 7. Use it

```cpp
#include "eoslab/core/core_calculations.hpp"

glis::eos::EoS eos{MyIdeal<2>{...}, MyResidual<2>{...}};

std::array<double, 2> x{0.4, 0.6};
std::span<const double, 2> xs{x};
double p = glis::eos::calc_pressure(eos, /*c=*/100.0, xs, /*T=*/300.0);
```

## 8. Test it

[`tests/derivative_test_harness.hpp`](../tests/derivative_test_harness.hpp)
provides reusable, model-agnostic checks. Drop a `.cpp` file in `tests/` (it is
auto-detected and registered by CMake) and compose them:

```cpp
#include "derivative_test_harness.hpp"
using namespace eoslab_test;

// 1. Every derivative-based property vs a 4th-order finite-difference reference
//    (computed in long double). Run several states, single-component AND mixture.
"my model derivative consistency"_test = [] {
    auto eos = glis::eos::EoS{MyIdeal<2>{...}, glis::eos::NoResidual<2>{}};
    run_derivative_consistency_tests<2>(eos, /*c=*/100.0, {0.4, 0.6}, /*T=*/300.0);
};

// 2. Structural Helmholtz consistency: Psi == c*a and Psi == sum_i Psi_i.
"my model helmholtz consistency"_test = [] {
    MyIdeal<2> model{...};
    check_helmholtz_consistency<2>(model, /*rho=*/{40.0, 70.0}, /*T=*/320.0);
};

// 3. Ideal-gas pressure p == c R T (ideal models paired with NoResidual).
"my ideal gas pressure"_test = [] {
    auto eos = glis::eos::EoS{MyIdeal<2>{...}, glis::eos::NoResidual<2>{}};
    check_ideal_gas_pressure<2>(eos, /*c=*/80.0, {0.35, 0.65}, /*T=*/330.0);
};

// 4. Euler relation: sum_i rho_i mu_i - Psi == calc_pressure (any EoS).
"my model euler pressure"_test = [] {
    auto eos = glis::eos::EoS{MyIdeal<2>{...}, MyResidual<2>{...}};
    check_euler_pressure<2>(eos, /*rho=*/{36.0, 54.0}, /*T=*/360.0);
};
```

`run_derivative_consistency_tests` validates the autodiff against finite
differences of your *own* Helmholtz functions, so it catches autodiff/wiring
mistakes — but it cannot tell you the Helmholtz energy itself is physically
correct (a wrong formula passes if its derivatives are self-consistent). Always
add model-specific checks against values you know analytically: reference-state
properties, ideal-gas / virial limits, etc. See
[`tests/test_const_cp.cpp`](../tests/test_const_cp.cpp) (caloric properties
against the reference data) and the closed-form checks in
[`tests/test_core_calculations.cpp`](../tests/test_core_calculations.cpp).

**Run the suite in `debug` as well as `release`** — the `-O1` debug build is what
surfaces a missing `GLIS_EOS_ALWAYS_INLINE` (§5).

## 9. Build / Enzyme requirements

EoSLab depends on Enzyme, which is a Clang/LLVM plugin. Therefore:

- **Clang is required** (CMake enforces this and defaults to `clang++`).
- **Optimization is required.** Enzyme runs inside the optimization pipeline and
  produces wrong derivatives at `-O0`. The `debug` preset uses `-O1 -g`;
  `release` uses `-O3`. Note that `-O1` alone is *not* sufficient for correct
  derivatives through the CRTP bases — the `GLIS_EOS_ALWAYS_INLINE` on the kernels
  / pre-calculations (§5) is what makes it correct there. Do not remove it.
- **UndefinedBehaviorSanitizer is incompatible** with Enzyme (its
  `__ubsan_handle_*` runtime calls cannot be differentiated). Only
  AddressSanitizer is enabled by the sanitizer option.
- Make your kernels **templated** on the floating-point type and use standard
  math functions (`std::log`, `std::exp`, `detail::fast_pow`, …); Enzyme
  differentiates these, and the test harness needs a `long double` instantiation.

```sh
cmake --preset release && cmake --build build/release && ctest --test-dir build/release
cmake --preset debug   && cmake --build build/debug   && ctest --test-dir build/debug
```
