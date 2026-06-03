#pragma once

#include <concepts>
namespace glis::eos {

template<std::floating_point Number = double> inline constexpr Number ideal_gas_constant{8.314462618};

}