#pragma once

#include "eoslab/core/parameter_storage.hpp"

#include <concepts>
#include <cstddef>
#include <span>

namespace glis::eos {

namespace detail {
/// @internal Does @p D define a pre-calculation for the (c, x, T) signature?
template<class D, class Number>
concept has_molar_pre_calc = requires(const D& d, Number c, const Number* x, Number T) {
    d.perform_pre_calculations(c, x, T);
};
/// @internal Does @p D define a pre-calculation for the (rho_i, T) signature?
template<class D, class Number>
concept has_density_pre_calc = requires(const D& d, const Number* rho_i, Number T) {
    d.perform_pre_calculations(rho_i, T);
};
} // namespace detail

template<class Derived, class ParamStruct, std::size_t N = std::dynamic_extent>
class MultiFluidBase : public ParameterStorage<ParamStruct, N> {
public:
    using parameter_type = ParamStruct;

    using ParameterStorage<ParamStruct, N>::ParameterStorage;

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz(Number c, const Number* x, Number T) const
    {
        Number a{0};
        if constexpr (detail::has_molar_pre_calc<Derived, Number>) {
            const auto pre = self().perform_pre_calculations(c, x, T);
            this->for_each_component(
                [&](std::size_t i) { a += self().calc_helmholtz_i(c, x, T, i, this->get_parameters(i), pre); });
        }
        else {
            this->for_each_component(
                [&](std::size_t i) { a += self().calc_helmholtz_i(c, x, T, i, this->get_parameters(i)); });
        }
        return a;
    }

    template<std::floating_point Number>
    [[nodiscard]] Number calc_helmholtz_density(const Number* rho_i, Number T) const
    {
        Number psi{0};
        if constexpr (detail::has_density_pre_calc<Derived, Number>) {
            const auto pre = self().perform_pre_calculations(rho_i, T);
            this->for_each_component([&](std::size_t i) {
                psi += self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i), pre);
            });
        }
        else {
            this->for_each_component(
                [&](std::size_t i) { psi += self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i)); });
        }
        return psi;
    }

    template<std::floating_point Number>
    void calc_partial_helmholtz(const Number* rho_i, Number T, Number* out) const
    {
        if constexpr (detail::has_density_pre_calc<Derived, Number>) {
            const auto pre = self().perform_pre_calculations(rho_i, T);
            this->for_each_component([&](std::size_t i) {
                out[i] = self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i), pre);
            });
        }
        else {
            this->for_each_component(
                [&](std::size_t i) { out[i] = self().calc_helmholtz_density_i(rho_i, T, i, this->get_parameters(i)); });
        }
    }

protected:
    [[nodiscard]] const Derived& self() const { return static_cast<const Derived&>(*this); }
};

} // namespace glis::eos
