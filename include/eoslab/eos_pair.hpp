#pragma once

#include "eoslab/concepts.hpp"
namespace glis::eos {
template<IdealEoS Ideal, ResidualEoS Residual> class EoS {
public:
    using ideal_type = Ideal;
    using residual_type = Residual;

    EoS(Ideal ideal, Residual residual) noexcept(std::is_nothrow_move_constructible_v<Ideal> &&
                                                 std::is_nothrow_move_constructible_v<Residual>) :
        ideal_{std::move(ideal)}, residual_{std::move(residual)}
    {
        assert(ideal_.size() == residual_.size());
    }

    // TODO: make these nodiscard?
    const Ideal& ideal() const noexcept { return ideal_; }
    const Residual& residual() const noexcept { return residual_; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return ideal_.size(); }

private:
    Ideal ideal_;
    Residual residual_;
};
} // namespace glis::eos