#pragma once

/**
 * @file eoslab.hpp
 * @brief Umbrella header for EoSLab; including it pulls in the public API.
 *
 * This is the single header in the root @c include/eoslab directory. It
 * aggregates the core machinery (the EoS pair, concepts, and thermodynamic
 * property calculations) together with the bundled models so that a translation
 * unit can access the whole library with one include:
 *
 * @code
 * #include "eoslab/eoslab.hpp"
 * @endcode
 *
 * Individual components remain available under their respective subdirectories
 * (@c core/, @c ideal_models/, @c residual_models/) for finer-grained includes.
 */

#include "eoslab/core/core_calculations.hpp"
#include "eoslab/core/xlnx.hpp"
#include "eoslab/ideal_models/const_cp.hpp"
#include "eoslab/ideal_models/nasa7.hpp"
#include "eoslab/ideal_models/nasa9.hpp"
#include "eoslab/residual_models/no_residual.hpp"
#include "eoslab/residual_models/peng_robinson.hpp"
#include "eoslab/residual_models/van_der_waals.hpp"
