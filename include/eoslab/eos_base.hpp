#pragma once

#include <cassert>
#include <cstddef>
#include <span>
namespace glis::eos {
// these provide base class with the needed information regarding compile time and runtime known sizes
template<std::size_t N = std::dynamic_extent> class BaseEoS {
public:
    constexpr BaseEoS() noexcept = default;
    constexpr explicit BaseEoS(const std::size_t n) { assert(n == N); }
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
};

template<> class BaseEoS<std::dynamic_extent> {
public:
    constexpr explicit BaseEoS(const std::size_t n) noexcept : n_(n) {}
    [[nodiscard]] constexpr std::size_t size() const noexcept { return n_; }

private:
    std::size_t n_;
};

// this is just used to help enforce correctness in the IdealEoS and ResidualEoS concepts
class BaseIdealEoS {};
} // namespace glis::eos