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

// Per-component decomposition of the Helmholtz energy DENSITY Psi.
//   rho_i : partial-molar-concentration array [mol/m^3]
//   T     : temperature                       [K]
//   out   : output array; out[i] = per-component Helmholtz density [J/m^3]
//           such that sum_i out[i] = Psi(rho, T)
template<std::floating_point Number>
void calc_partial_helmholtz(const Number* rho_i, Number T, Number* out) const;

// Number of components [-].
std::size_t size() const;   // (provided for you by BaseEoS<N>)
```

> **Mind the units.** `calc_helmholtz` returns an *intensive molar* energy
> [J/mol], whereas `calc_partial_helmholtz` returns a *volumetric density*
> [J/m³]. They must be consistent:
>
> $$
> \Psi(\vec\rho,T) = c a(c,\vec{x},T) \quad \text{with} \quad c = \sum_i \rho_i,\quad x_i=\frac{\rho_i}{c}
> $$
>
> Implement both from a single definition so this relationship holds exactly.

The easiest way to get `size()` (and the component-count plumbing) is to derive
from `glis::eos::BaseEoS<N>` ([`eos_base.hpp`](../include/eoslab/core/eos_base.hpp)).
Use a concrete `N` for a compile-time component count, or `std::dynamic_extent`
for a runtime one.

## 3. Requirements for an `IdealEoS`

```cpp
template<class E>
concept IdealEoS = std::derived_from<E, BaseIdealEoS> && EquationOfState<E>;
```

An ideal model must:

1. **Derive publicly from `glis::eos::BaseIdealEoS`** (the tag that marks it as
   ideal), and
2. satisfy `EquationOfState` (provide the three members above).

Additionally, for the property routines to be physically consistent, a genuine
ideal gas must have compressibility factor `Z = 1`, which means its molar
Helmholtz energy depends on concentration as

$$
\left(\frac{\partial a_\text{ideal}}{\partial c}\right)_{T,\vec x} = \frac{RT}{c} \iff a_\text{ideal} = RT \ln c + f(T,\vec x)
$$

EoSLib makes this assumption internally to simplify calculations. The reference for the derivative calculations we implemented may be found [here](https://doi.org/10.1021/acs.iecr.2c00237).

## 4. Requirements for a `ResidualEoS`

```cpp
template<class E>
concept ResidualEoS = EquationOfState<E> && !IdealEoS<E>;
```

A residual model must satisfy `EquationOfState` and **must not** derive from
`BaseIdealEoS`. That is the only thing distinguishing it from an ideal model at
compile time. Its `calc_helmholtz` returns the *residual* molar Helmholtz energy
(the departure from ideal-gas behaviour), and `alpha_res -> 0` as `c -> 0`.

## 5. Add a `static_assert` (recommended)

Right below your model, assert that it satisfies the concept you intend — this
turns a subtle interface mistake (wrong signature, missing `const`, forgetting
to derive from `BaseIdealEoS`, …) into an immediate, readable compile error:

```cpp
#include "eoslab/core/concepts.hpp"

template<std::size_t N>
class MyIdealModel : public glis::eos::BaseIdealEoS, public glis::eos::BaseEoS<N> {
    // ... calc_helmholtz / calc_partial_helmholtz ...
};
static_assert(glis::eos::IdealEoS<MyIdealModel<2>>,
              "MyIdealModel must satisfy the IdealEoS concept");

template<std::size_t N>
class MyResidualModel : public glis::eos::BaseEoS<N> {
    // ... calc_helmholtz / calc_partial_helmholtz ...
};
static_assert(glis::eos::ResidualEoS<MyResidualModel<2>>,
              "MyResidualModel must satisfy the ResidualEoS concept");
```

See [`tests/eos_test_models.hpp`](../tests/eos_test_models.hpp) for two complete,
worked examples (a constant-`c_v` ideal gas and a truncated-virial residual), and
[`tests/test_concepts.cpp`](../tests/test_concepts.cpp) for concept assertions.

## 6. Use it

```cpp
#include "eoslab/core/core_calculations.hpp"

glis::eos::EoS eos{MyIdealModel<2>{...}, MyResidualModel<2>{...}};

std::array<double, 2> x{0.4, 0.6};
std::span<const double, 2> xs{x};
double p = glis::eos::calc_pressure(eos, /*c=*/100.0, xs, /*T=*/300.0);
```

## 7. Test it

A templated finite-difference harness in
[`tests/derivative_test_harness.hpp`](../tests/derivative_test_harness.hpp)
checks **every** derivative-based property of an arbitrary EoS pair against an
independent 4th-order central finite-difference reference (computed in
`long double`). To add coverage for your model, drop a `.cpp` file in `tests/`
(it is auto-detected and registered by CMake) and call:

```cpp
#include "derivative_test_harness.hpp"
using namespace eoslab_test;

"my model derivative consistency"_test = [] {
    glis::eos::EoS eos{MyIdealModel<2>{...}, MyResidualModel<2>{...}};
    run_derivative_consistency_tests<2>(eos, /*c=*/100.0, {0.4, 0.6}, /*T=*/300.0);
};
```

Prefer analytic references when you know them (see the closed-form checks in
[`tests/test_core_calculations.cpp`](../tests/test_core_calculations.cpp)); fall
back to the finite-difference harness otherwise.

## 8. Build / Enzyme requirements

EoSLab depends on Enzyme, which is a Clang/LLVM plugin. Therefore:

- **Clang is required** (CMake enforces this and defaults to `clang++`).
- **Optimization is required.** Enzyme runs inside the optimization pipeline and
  produces wrong derivatives at `-O0` (reverse mode in particular). The `debug`
  preset uses `-O1 -g` for this reason; `release` uses `-O3`.
- **UndefinedBehaviorSanitizer is incompatible** with Enzyme (its
  `__ubsan_handle_*` runtime calls cannot be differentiated). Only
  AddressSanitizer is enabled by the sanitizer option.
- Make your `calc_helmholtz` / `calc_partial_helmholtz` **templated** on the
  floating-point type and use standard math functions (`std::log`, `std::exp`,
  `detail::fast_pow`, …); Enzyme differentiates these, and the test harness needs
  a `long double` instantiation.

```sh
cmake --preset release && cmake --build build/release && ctest --test-dir build/release
```
