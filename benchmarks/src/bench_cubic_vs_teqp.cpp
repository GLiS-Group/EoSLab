//
// Head-to-head benchmark of this library's cubic residual models against
// NIST's teqp (https://github.com/usnistgov/teqp).
//
// What is compared
// ----------------
// The dimensionless reduced residual Helmholtz derivatives
//
//     Ar_{ij} = (1/T)^i rho^j d^{i+j} alphar / d(1/T)^i d rho^j
//
// which teqp exposes as TDXDerivatives::get_Arxy<i, j> (get_Ar00, get_Ar10,
// get_Ar01, get_Ar02, ...) and this library as detail::calc_lambda<i, j>
// applied to the residual model. Both sides use the same species data and the
// same state, so the rows labelled .../EoSLab and .../teqp time the same
// mathematical quantity.
//
// Layout: van der Waals first, then Peng-Robinson. Within each model family
// the species count N sweeps {1, 5, 10, 50, 100, 500, 1000}; for each N every
// derivative is benchmarked as the pair (EoSLab row, then teqp row).
//
// This library's models are used in their runtime-sized (dynamic-extent)
// form, matching teqp's runtime-sized models; the parameter matrices then
// live on the heap, which the N = 1000 case requires (an N x N double matrix
// is 8 MB). kij = 0 on both sides.
//
#include "eoslab/core/core_calculations.hpp"
#include "eoslab/residual_models/peng_robinson.hpp"
#include "eoslab/residual_models/van_der_waals.hpp"

#include "teqp/derivs.hpp"
#include "teqp/models/cubics.hpp"
#include "teqp/models/vdW.hpp"

#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <valarray>
#include <vector>

namespace ge = glis::eos;

namespace {

// ---------------------------------------------------------------------------
// Deterministic per-species critical data shared by both libraries.
// ---------------------------------------------------------------------------
struct Crit {
    double T_c;   // [K]
    double P_c;   // [Pa]
    double omega; // [-]
};

std::vector<Crit> make_crit(std::size_t n)
{
    std::mt19937_64 gen(n);
    std::uniform_real_distribution<double> dTc(120.0, 600.0);
    std::uniform_real_distribution<double> dPc(1.0e6, 8.0e6);
    std::uniform_real_distribution<double> dw(0.0, 0.45);
    std::vector<Crit> crit(n);
    for (Crit& c : crit) {
        c = {.T_c = dTc(gen), .P_c = dPc(gen), .omega = dw(gen)};
    }
    return crit;
}

ge::VanDerWaals<> make_mine_vdw(const std::vector<Crit>& crit)
{
    using SI = ge::VanDerWaals<>::SpeciesInput;
    std::vector<SI> in;
    in.reserve(crit.size());
    for (const Crit& c : crit) {
        in.push_back({.T_c = c.T_c, .P_c = c.P_c});
    }
    return ge::VanDerWaals<>(std::span<const SI>{in});
}

ge::PengRobinson<> make_mine_pr(const std::vector<Crit>& crit)
{
    using SI = ge::PengRobinson<>::SpeciesInput;
    std::vector<SI> in;
    in.reserve(crit.size());
    for (const Crit& c : crit) {
        in.push_back({.T_c = c.T_c, .P_c = c.P_c, .omega = c.omega});
    }
    return ge::PengRobinson<>(std::span<const SI>{in});
}

teqp::vdWEOS<double> make_teqp_vdw(const std::vector<Crit>& crit)
{
    const std::size_t n = crit.size();
    std::valarray<double> Tc(n);
    std::valarray<double> pc(n);
    for (std::size_t i = 0; i < n; ++i) {
        Tc[i] = crit[i].T_c;
        pc[i] = crit[i].P_c;
    }
    return {Tc, pc};
}

auto make_teqp_pr(const std::vector<Crit>& crit)
{
    const std::size_t n = crit.size();
    std::valarray<double> Tc(n);
    std::valarray<double> pc(n);
    std::valarray<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        Tc[i] = crit[i].T_c;
        pc[i] = crit[i].P_c;
        w[i] = crit[i].omega;
    }
    return teqp::canonical_PR(Tc, pc, w);
}

// ---------------------------------------------------------------------------
// Bench: one model from each library plus the shared state. The composition is
// a deterministic non-uniform mixture; the state is a moderate-density gas
// (b_m * c stays well below 1 for the species ranges above).
// ---------------------------------------------------------------------------
template<class Mine, class Teqp> struct Bench {
    Mine mine;
    Teqp tq;
    double c = 500.0; // molar concentration [mol/m^3]
    double T = 350.0; // temperature [K]
    std::vector<double> x;
    Eigen::ArrayXd z;

    Bench(Mine m, Teqp t, std::size_t n) : mine(std::move(m)), tq(std::move(t)), x(n), z(n)
    {
        std::mt19937_64 gen(n);
        std::uniform_real_distribution<double> frac(0.2, 1.0);
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = frac(gen);
            sum += x[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            x[i] /= sum;
            z[static_cast<Eigen::Index>(i)] = x[i];
        }
    }
};

// ---------------------------------------------------------------------------
// Register the comparable pair of rows for one derivative order (iT, iD):
// first this library (Enzyme), then teqp (autodiff).
// ---------------------------------------------------------------------------
template<int iT, int iD, class B> void register_pair(const std::string& name, const std::shared_ptr<B>& b)
{
    benchmark::RegisterBenchmark(name + "/EoSLab", [b](benchmark::State& st) {
        for (auto _ : st) {
            benchmark::DoNotOptimize(b->c);
            benchmark::DoNotOptimize(b->T);
            const double invT = 1.0 / b->T;
            auto v = ge::detail::calc_lambda<iT, iD>(b->mine, b->c, b->x.data(), invT);
            benchmark::DoNotOptimize(v);
        }
    });
    benchmark::RegisterBenchmark(name + "/teqp", [b](benchmark::State& st) {
        using tdx = teqp::TDXDerivatives<std::decay_t<decltype(b->tq)>, double, Eigen::ArrayXd>;
        for (auto _ : st) {
            benchmark::DoNotOptimize(b->c);
            benchmark::DoNotOptimize(b->T);
            if constexpr (iT == 0 && iD == 0) {
                auto v = tdx::get_Ar00(b->tq, b->T, b->c, b->z);
                benchmark::DoNotOptimize(v);
            }
            else {
                auto v = tdx::template get_Arxy<iT, iD>(b->tq, b->T, b->c, b->z);
                benchmark::DoNotOptimize(v);
            }
        }
    });
}

template<class B> void register_all_derivs(const std::string& prefix, const std::shared_ptr<B>& b)
{
    register_pair<0, 0>(prefix + "/Ar00", b);
    register_pair<1, 0>(prefix + "/Ar10", b);
    register_pair<0, 1>(prefix + "/Ar01", b);
    register_pair<0, 2>(prefix + "/Ar02", b);
    register_pair<1, 1>(prefix + "/Ar11", b);
    register_pair<2, 0>(prefix + "/Ar20", b);
}

constexpr std::array<std::size_t, 7> kSizes{1, 5, 10, 50, 100, 500, 1000};

void register_vdw(std::size_t n)
{
    const std::vector<Crit> crit = make_crit(n);
    using B = Bench<ge::VanDerWaals<>, teqp::vdWEOS<double>>;
    auto b = std::make_shared<B>(make_mine_vdw(crit), make_teqp_vdw(crit), n);
    register_all_derivs("vdW/N" + std::to_string(n), b);
}

void register_pr(std::size_t n)
{
    const std::vector<Crit> crit = make_crit(n);
    using B = Bench<ge::PengRobinson<>, decltype(make_teqp_pr(crit))>;
    auto b = std::make_shared<B>(make_mine_pr(crit), make_teqp_pr(crit), n);
    register_all_derivs("PR/N" + std::to_string(n), b);
}

// Benchmarks run in registration order: vdW over all sizes, then PR.
void register_all()
{
    for (const std::size_t n : kSizes) {
        register_vdw(n);
    }
    for (const std::size_t n : kSizes) {
        register_pr(n);
    }
}

} // namespace

int main(int argc, char** argv)
{
    register_all();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
