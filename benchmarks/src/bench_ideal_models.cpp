//
// Benchmarks for the ideal-gas models in include/eoslab/ideal_models, exercised
// through every (non-detail) thermodynamic routine in core_calculations.hpp.
//
// What is covered
// ---------------
//   * Both bundled ideal models (ConstantCp and Nasa7), each in its
//     compile-time-sized ("Static") and runtime-sized ("Dynamic") form.
//   * Every public core calculation.
//
// Each ideal model is paired with NoResidual, so the EoS is a pure ideal gas.
// Consequently the residual-only properties (pressure, dp_dc, dp_dT) are cheap
// (their residual derivatives vanish); they are still benchmarked for
// completeness. The caloric properties and the chemical-potential gradients are
// where the ideal model's autodiff is actually exercised.
//
// Output layout
// -------------
// A custom driver groups the comparable rows together. The nesting is
//     size N  ->  calculation  ->  model family  ->  { static, dynamic }
// so each calculation produces a block of two rows -- static, dynamic -- which
// is the natural unit to read. A large banner marks each size and a small
// heading marks each calculation/family block.
//
#include "eoslab/core/core_calculations.hpp"
#include "eoslab/core/eos_pair.hpp"
#include "eoslab/ideal_models/const_cp.hpp"
#include "eoslab/ideal_models/nasa7.hpp"
#include "eoslab/residual_models/no_residual.hpp"

#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ge = glis::eos;

namespace {

// ---------------------------------------------------------------------------
// span helpers: build a span of the right (static or dynamic) extent E.
// ---------------------------------------------------------------------------
template<std::size_t E> auto cspan(const std::vector<double>& v)
{
    if constexpr (E == std::dynamic_extent) {
        return std::span<const double>{v.data(), v.size()};
    }
    else {
        return std::span<const double, E>{v.data(), E};
    }
}
template<std::size_t E> auto mspan(std::vector<double>& v)
{
    if constexpr (E == std::dynamic_extent) {
        return std::span<double>{v.data(), v.size()};
    }
    else {
        return std::span<double, E>{v.data(), E};
    }
}

// ---------------------------------------------------------------------------
// Deterministic per-species inputs. The two model families have different
// SpeciesInput layouts; pick the right one by detecting a NASA-7 field.
// ---------------------------------------------------------------------------
template<class SI>
concept nasa_input = requires(SI s) { s.a0; };

template<class SI, class Gen, class Dist> SI make_input(Gen& g, Dist& d)
{
    if constexpr (nasa_input<SI>) {
        return SI{.a0 = 3.5 * d(g),
                  .a1 = 1.0e-3 * d(g),
                  .a2 = 1.0e-5 * d(g),
                  .a3 = 1.0e-8 * d(g),
                  .a4 = 1.0e-12 * d(g),
                  .a5 = -1.0e3 * d(g),
                  .a6 = 3.0 * d(g),
                  .T_ref = 298.15,
                  .p_ref = 1.0e5};
    }
    else {
        return SI{.T_ref = 298.15, .p_ref = 1.0e5, .c_p = 29.0 * d(g), .h_ref = 0.0, .s_ref = 130.0 * d(g)};
    }
}

template<template<std::size_t> class Tmpl, std::size_t N> Tmpl<N> make_model(std::size_t n)
{
    using Model = Tmpl<N>;
    using SI = typename Model::SpeciesInput;
    std::mt19937_64 gen(0xEAB0ULL + n);
    std::uniform_real_distribution<double> d(0.5, 2.0);
    if constexpr (N == std::dynamic_extent) {
        std::vector<SI> in;
        in.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            in.push_back(make_input<SI>(gen, d));
        }
        return Model(std::span<const SI>(in));
    }
    else {
        std::array<SI, N> in{};
        for (std::size_t i = 0; i < N; ++i) {
            in[i] = make_input<SI>(gen, d);
        }
        return Model(in);
    }
}

// ---------------------------------------------------------------------------
// Bench: an ideal model paired with NoResidual, plus the working buffers.
// ---------------------------------------------------------------------------
template<class Ideal, std::size_t N> struct Bench {
    using Residual = ge::NoResidual<N>;
    using Pair = ge::EoS<Ideal, Residual>;

    static constexpr std::size_t extent = N; ///< Span extent for the property calls.

    Pair eos;
    double c = 120.0;          // total molar concentration [mol/m^3]
    double T = 350.0;          // temperature [K]
    double molar_mass = 0.029; // effective molar mass [kg/mol]
    std::vector<double> x;     // mole fractions (sum to 1)
    std::vector<double> rho;   // partial concentrations (= x * c)
    std::vector<double> out;   // scratch output for the gradient routines

    Bench(std::size_t n, Ideal ideal) : eos(make_pair(std::move(ideal), n)), x(n), rho(n), out(n)
    {
        std::mt19937_64 gen(0x5EED0ULL + n);
        std::uniform_real_distribution<double> frac(0.2, 1.0);
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = frac(gen);
            sum += x[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            x[i] /= sum;
            rho[i] = x[i] * c;
        }
    }

    static Pair make_pair(Ideal ideal, std::size_t n)
    {
        if constexpr (N == std::dynamic_extent) {
            return Pair{std::move(ideal), Residual(n)};
        }
        else {
            (void)n;
            return Pair{std::move(ideal), Residual{}};
        }
    }
};

// ---------------------------------------------------------------------------
// The set of core calculations, with the label used in the benchmark name.
// ---------------------------------------------------------------------------
enum class Calc {
    helmholtz,
    pressure,
    internal_energy,
    enthalpy,
    entropy,
    gibbs,
    dp_dc,
    dp_dT,
    cv,
    cp,
    sound_speed_sq,
    chemical_potential,
    log_fugacity_coeff,
    fugacity,
};

struct CalcInfo {
    Calc id;
    const char* name;
};

constexpr std::array<CalcInfo, 14> kCalcs{{
    {Calc::helmholtz, "helmholtz"},
    {Calc::pressure, "pressure"},
    {Calc::internal_energy, "internal_energy"},
    {Calc::enthalpy, "enthalpy"},
    {Calc::entropy, "entropy"},
    {Calc::gibbs, "gibbs"},
    {Calc::dp_dc, "dp_dc"},
    {Calc::dp_dT, "dp_dT"},
    {Calc::cv, "cv"},
    {Calc::cp, "cp"},
    {Calc::sound_speed_sq, "sound_speed_sq"},
    {Calc::chemical_potential, "chemical_potential"},
    {Calc::log_fugacity_coeff, "log_fugacity_coeff"},
    {Calc::fugacity, "fugacity"},
}};

// The size sweep used by the grouped default run.
constexpr std::array<std::size_t, 6> kSizes{1, 2, 10, 50, 100, 1000};

// Heap-allocated benches kept alive for the whole process.
std::vector<std::shared_ptr<void>> g_keepalive;

// ---------------------------------------------------------------------------
// Register a single (calculation, bench) benchmark under the given name.
// ---------------------------------------------------------------------------
template<class B> void register_calc(const std::string& name, B* b, Calc calc)
{
    constexpr std::size_t E = B::extent;

#define SCALAR(...)                                                                                                    \
    benchmark::RegisterBenchmark(name, [b](benchmark::State& st) {                                                     \
        for (auto _ : st) {                                                                                            \
            benchmark::DoNotOptimize(b->c);                                                                            \
            benchmark::DoNotOptimize(b->T);                                                                            \
            auto v = (__VA_ARGS__);                                                                                    \
            benchmark::DoNotOptimize(v);                                                                               \
        }                                                                                                              \
    })
#define VOIDC(...)                                                                                                     \
    benchmark::RegisterBenchmark(name, [b](benchmark::State& st) {                                                     \
        for (auto _ : st) {                                                                                            \
            benchmark::DoNotOptimize(b->T);                                                                            \
            __VA_ARGS__;                                                                                               \
            benchmark::DoNotOptimize(b->out.data());                                                                   \
            benchmark::ClobberMemory();                                                                                \
        }                                                                                                              \
    })

    switch (calc) {
    case Calc::helmholtz:
        SCALAR(ge::calc_helmholtz(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::pressure:
        SCALAR(ge::calc_pressure(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::internal_energy:
        SCALAR(ge::calc_internal_energy(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::enthalpy:
        SCALAR(ge::calc_enthalpy(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::entropy:
        SCALAR(ge::calc_entropy(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::gibbs:
        SCALAR(ge::calc_gibbs(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::dp_dc:
        SCALAR(ge::calc_dp_dc(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::dp_dT:
        SCALAR(ge::calc_dp_dT(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::cv:
        SCALAR(ge::calc_cv(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::cp:
        SCALAR(ge::calc_cp(b->eos, b->c, cspan<E>(b->x), b->T));
        break;
    case Calc::sound_speed_sq:
        SCALAR(ge::calc_sound_speed_squared(b->eos, b->c, cspan<E>(b->x), b->T, b->molar_mass));
        break;
    case Calc::chemical_potential:
        VOIDC(ge::calc_chemical_potential(b->eos, cspan<E>(b->rho), b->T, mspan<E>(b->out)));
        break;
    case Calc::log_fugacity_coeff:
        VOIDC(ge::calc_log_fugacity_coeff(b->eos, b->c, cspan<E>(b->x), b->T, cspan<E>(b->rho), mspan<E>(b->out)));
        break;
    case Calc::fugacity:
        VOIDC(ge::calc_fugacity(b->eos, cspan<E>(b->rho), b->T, mspan<E>(b->out)));
        break;
    }

#undef SCALAR
#undef VOIDC
}

// ---------------------------------------------------------------------------
// Register both model families for one compile-time size. The two comparable
// rows of each calculation block are registered consecutively in the order
// static, dynamic so a single filtered run prints them in that order.
// ---------------------------------------------------------------------------
template<std::size_t N> void register_size()
{
    constexpr std::size_t dyn = std::dynamic_extent;

    auto cc_s = std::make_shared<Bench<ge::ConstantCp<N>, N>>(N, make_model<ge::ConstantCp, N>(N));
    auto cc_d = std::make_shared<Bench<ge::ConstantCp<dyn>, dyn>>(N, make_model<ge::ConstantCp, dyn>(N));
    auto n7_s = std::make_shared<Bench<ge::Nasa7<N>, N>>(N, make_model<ge::Nasa7, N>(N));
    auto n7_d = std::make_shared<Bench<ge::Nasa7<dyn>, dyn>>(N, make_model<ge::Nasa7, dyn>(N));

    g_keepalive.push_back(cc_s);
    g_keepalive.push_back(cc_d);
    g_keepalive.push_back(n7_s);
    g_keepalive.push_back(n7_d);

    const std::string ns = std::to_string(N);
    for (const CalcInfo& ci : kCalcs) {
        const std::string suffix = "/N" + ns + "/";
        // ConstantCp family: the two comparable rows, in display order.
        register_calc("Static" + suffix + "ConstantCp/" + ci.name, cc_s.get(), ci.id);
        register_calc("Dynamic" + suffix + "ConstantCp/" + ci.name, cc_d.get(), ci.id);
        // Nasa7 family: the two comparable rows, in display order.
        register_calc("Static" + suffix + "Nasa7/" + ci.name, n7_s.get(), ci.id);
        register_calc("Dynamic" + suffix + "Nasa7/" + ci.name, n7_d.get(), ci.id);
    }
}

void register_all()
{
    register_size<1>();
    register_size<5>();
    register_size<10>();
    register_size<50>();
    register_size<100>();
    register_size<500>();
    register_size<1000>();
    register_size<5000>();
}

// ---------------------------------------------------------------------------
// Reporter that prints the (verbose) run context only once, so the repeated
// per-block runs do not each re-dump the CPU/cache description.
// ---------------------------------------------------------------------------
class QuietReporter : public benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override
    {
        if (context_printed_) {
            return true;
        }
        context_printed_ = true;
        return benchmark::ConsoleReporter::ReportContext(context);
    }

private:
    bool context_printed_ = false;
};

void print_size_banner(std::size_t n)
{
    std::cout << "\n"
              << "########################################################################\n"
              << "##  N = " << n << "   (number of components)\n"
              << "########################################################################\n"
              << std::flush;
}

void print_calc_heading(const std::string& calc, const std::string& family, std::size_t n)
{
    std::cout << "\n---- " << calc << "  [" << family << ", N = " << n << "] "
              << "-- rows: static, dynamic\n"
              << std::flush;
}

} // namespace

int main(int argc, char** argv)
{
    register_all();

    // The custom driver below substitutes its own per-block filter, which would
    // otherwise shadow a user-supplied `--benchmark_filter`. Detect that flag up
    // front and, if present, fall back to a single plain run that honours it.
    bool user_filtered = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]).rfind("--benchmark_filter", 0) == 0) {
            user_filtered = true;
        }
    }

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    QuietReporter reporter;

    if (user_filtered) {
        // Initialize() has already applied the user's regex as the global filter.
        benchmark::RunSpecifiedBenchmarks(&reporter);
        benchmark::Shutdown();
        return 0;
    }

    constexpr std::array<const char*, 2> families{"ConstantCp", "Nasa7"};

    for (std::size_t n : kSizes) {
        print_size_banner(n);
        for (const CalcInfo& ci : kCalcs) {
            for (const char* family : families) {
                print_calc_heading(ci.name, family, n);
                // Match the two comparable rows for this (size, calc, family):
                // both phases (Static/Dynamic).
                const std::string spec = "/N" + std::to_string(n) + "/" + family + "/" + ci.name + "$";
                benchmark::RunSpecifiedBenchmarks(&reporter, spec);
            }
        }
    }

    benchmark::Shutdown();
    return 0;
}
