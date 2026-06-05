#pragma once

#include "eoslab/core/parameter_storage.hpp"

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace glis::eos {

template<class Derived, class ParamStruct, std::size_t N = std::dynamic_extent>
class MultiFluidBase : public ParameterStorage<ParamStruct, N> {
public:
    using parameter_type = ParamStruct;

    using ParameterStorage<ParamStruct, N>::ParameterStorage;

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz(Number /*c*/, const Number* /*x*/, Number /*T*/) const
    {
        throw std::runtime_error{"not implemented"};
    }

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density(const Number* /*rho_i*/, Number /*T*/) const
    {
        throw std::runtime_error{"not implemented"};
    }

    template<std::floating_point Number>
    void calc_partial_helmholtz(const Number* /*rho_i*/, Number /*T*/, Number* /*out*/) const
    {
        throw std::runtime_error{"not implemented"};
    }

protected:
    [[nodiscard]] const Derived& self() const { return static_cast<const Derived&>(*this); }
};

} // namespace glis::eos
