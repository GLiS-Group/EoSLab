#pragma once
/**
 * @file parameter_storage.hpp
 * @brief Structure-of-Arrays storage for an EoS model's per-species parameters.
 *
 * A model defines one meaningfully-named POD struct holding the parameters of a
 * single species (e.g. `struct Params { double h; double g; };`) and constructs
 * the storage from a list of those structs, one per species. Internally the data
 * is laid out column-major (each parameter contiguous across all species), which
 * keeps mixing-rule reductions and across-species vectorization cache- and
 * SIMD-friendly at large component counts. Implementers never touch flat
 * indices: they read back a named struct for one species (get_parameters) or a
 * contiguous span of one parameter across all species (column).
 */

#include "eoslab/core/aggregate.hpp"
#include "eoslab/core/eos_base.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace glis::eos {

/**
 * @brief SoA per-species parameter storage.
 *
 * @tparam ParamStruct Per-species parameter aggregate. Must be a trivially
 *         copyable, all-`double` aggregate (no padding); its member count is the
 *         number of parameters per species and is derived automatically.
 * @tparam N Component count, or @c std::dynamic_extent for a runtime count.
 */
template<class ParamStruct, std::size_t N = std::dynamic_extent> class ParameterStorage : public BaseEoS<N> {
public:
    using parameter_type = ParamStruct;                ///< The per-species parameter struct.
    static constexpr std::size_t parameter_count = detail::aggregate_arity<ParamStruct>(); ///< Parameters per species.

    /// @brief Construct with all parameters zero-initialized.
    constexpr ParameterStorage() = default;

    /**
     * @brief Construct from a per-species list of parameter structs.
     * @param rows One @p ParamStruct per species; transposed into column-major
     *             storage on construction.
     */
    constexpr explicit ParameterStorage(const std::array<ParamStruct, N>& rows)
    {
        for (std::size_t i = 0; i < N; ++i) {
            const std::array<double, parameter_count> members = detail::to_double_array(rows[i]);
            for (std::size_t p = 0; p < parameter_count; ++p) {
                data_[(p * N) + i] = members[p];
            }
        }
    }

    /**
     * @brief Parameters of species @p i as a named struct (row view).
     * @param i Species index; must be `< size()`.
     * @return The reconstructed @p ParamStruct for species @p i.
     */
    [[nodiscard]] [[gnu::always_inline]] ParamStruct get_parameters(std::size_t i) const
    {
        assert(i < N);
        return detail::make_from_indexed<ParamStruct>([&](std::size_t p) { return data_[(p * N) + i]; });
    }

    /**
     * @brief Contiguous view of parameter @p p across all species (column view).
     * @param p Parameter index; must be `< parameter_count`.
     * @return A span of length `size()` over parameter @p p for every species.
     */
    [[nodiscard]] std::span<const double, N> column(std::size_t p) const
    {
        assert(p < parameter_count);
        return std::span<const double, N>{data_.data() + (p * N), N};
    }

private:
    std::array<double, parameter_count * N> data_{};
};

/**
 * @brief Runtime-sized specialization of ParameterStorage.
 *
 * Stores the columns in a @c std::vector because the component count is not
 * known until construction.
 */
template<class ParamStruct>
class ParameterStorage<ParamStruct, std::dynamic_extent> : public BaseEoS<std::dynamic_extent> {
public:
    using parameter_type = ParamStruct;                ///< The per-species parameter struct.
    static constexpr std::size_t parameter_count = detail::aggregate_arity<ParamStruct>(); ///< Parameters per species.

    /**
     * @brief Construct from a per-species list of parameter structs.
     * @param rows One @p ParamStruct per species; `size()` becomes `rows.size()`
     *             and the data is transposed into column-major storage.
     */
    explicit ParameterStorage(std::span<const ParamStruct> rows) : BaseEoS<std::dynamic_extent>(rows.size())
    {
        const std::size_t n = rows.size();
        data_.resize(parameter_count * n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::array<double, parameter_count> members = detail::to_double_array(rows[i]);
            for (std::size_t p = 0; p < parameter_count; ++p) {
                data_[(p * n) + i] = members[p];
            }
        }
    }

    /**
     * @brief Parameters of species @p i as a named struct (row view).
     * @param i Species index; must be `< size()`.
     * @return The reconstructed @p ParamStruct for species @p i.
     */
    [[nodiscard]] [[gnu::always_inline]] ParamStruct get_parameters(std::size_t i) const
    {
        assert(i < this->size());
        const std::size_t n = this->size();
        return detail::make_from_indexed<ParamStruct>([&](std::size_t p) { return data_[(p * n) + i]; });
    }

    /**
     * @brief Contiguous view of parameter @p p across all species (column view).
     * @param p Parameter index; must be `< parameter_count`.
     * @return A span of length `size()` over parameter @p p for every species.
     */
    [[nodiscard]] std::span<const double> column(std::size_t p) const
    {
        assert(p < parameter_count);
        const std::size_t n = this->size();
        return std::span<const double>{data_.data() + (p * n), n};
    }

private:
    std::vector<double> data_;
};

} // namespace glis::eos
