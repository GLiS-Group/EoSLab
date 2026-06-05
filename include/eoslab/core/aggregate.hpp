#pragma once
/**
 * @file aggregate.hpp
 * @brief Compile-time reflection helpers for simple aggregate (POD) structs.
 *
 * Lets the library work with implementer-defined parameter / pre-calculation
 * structs by their member *count* and *order* without the implementer having to
 * restate that count anywhere: aggregate_arity() recovers the number of members,
 * and make_from_indexed() builds an aggregate from a per-member accessor. These
 * are the pre-C++26-reflection idioms; if the project adopts C++26 reflection,
 * this is the only header that needs to change.
 */

#include <array>
#include <bit>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace glis::eos::detail {

/// @cond INTERNAL

// Probe type whose templated conversion operator stands in for "any member" when
// counting how many braced initializers an aggregate accepts. Used only in
// unevaluated contexts; intentionally left without a definition. The constraint
// keeps it from converting to itself, which avoids copy/move ambiguity in the
// single-field case.
struct any_field {
    template<class T>
        requires(!std::is_same_v<T, any_field>)
    constexpr operator T() const;
};

// any_field regardless of the index it is generated from (used to expand a pack
// of N probe initializers).
template<std::size_t> using any_field_t = any_field;

// Is T aggregate-initializable from exactly sizeof...(Is) probe initializers?
template<class T, std::size_t... Is>
concept brace_constructible_with = requires { T{any_field_t<Is>{}...}; };

// Is T aggregate-initializable from exactly N probe initializers?
template<class T, std::size_t N> [[nodiscard]] consteval bool constructible_with_n()
{
    return []<std::size_t... Is>(std::index_sequence<Is...>) { return brace_constructible_with<T, Is...>; }(
        std::make_index_sequence<N>{});
}

/// @endcond

/**
 * @brief Number of data members of an aggregate type.
 *
 * Determines, at compile time, how many initializers @p T accepts in
 * aggregate initialization &mdash; i.e. its member count (its "arity").
 *
 * @tparam T An aggregate type (no user-declared/inherited constructors, no base
 *           classes) whose members are flat (no nested-brace-elided aggregates).
 *           Up to 64 members are supported.
 * @return The number of data members of @p T (0 for an empty aggregate).
 *
 * @code{.cpp}
 * struct Params { double h; double g; };
 * static_assert(aggregate_arity<Params>() == 2);
 * @endcode
 */
template<class T> [[nodiscard]] consteval std::size_t aggregate_arity()
{
    static_assert(std::is_aggregate_v<T>, "aggregate_arity<T>: T must be an aggregate type.");
    // An aggregate accepts 0, 1, ..., arity initializers (extra members are
    // value-initialized) and rejects more. Counting how many of those counts are
    // valid therefore gives arity + 1; subtract the empty-braces (N == 0) case.
    constexpr std::size_t cap = 64;
    const std::size_t valid_counts = []<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((constructible_with_n<T, Is>() ? std::size_t{1} : std::size_t{0}) + ... + std::size_t{0});
    }(std::make_index_sequence<cap + 1>{});
    return valid_counts - 1;
}

/**
 * @brief Variable-template spelling of aggregate_arity() for use in template
 *        arguments and array sizes.
 * @tparam T An aggregate type (see aggregate_arity()).
 */
template<class T> inline constexpr std::size_t aggregate_arity_v = aggregate_arity<T>();

/**
 * @brief Construct an aggregate from a per-member accessor.
 *
 * Builds `T{ access(0), access(1), ..., access(arity-1) }`, where the arity is
 * deduced via aggregate_arity(). Initializers are evaluated left-to-right, so
 * members are filled in declaration order. Intended for homogeneous aggregates
 * (e.g. all-`double` parameter structs); a mixed-type @p T may reject a uniform
 * accessor on narrowing grounds.
 *
 * @tparam T        The aggregate type to build.
 * @tparam Accessor Callable invocable as `access(std::size_t)`, returning a value
 *                  convertible (without narrowing) to each member type.
 * @param  access   Accessor invoked once per member, with the member's index.
 *                  Taken by value; may carry state.
 * @return A @p T aggregate-initialized from the accessor results in member order.
 *
 * @code{.cpp}
 * struct Params { double h; double g; };
 * // Reconstruct species i from column-major storage:
 * Params p = make_from_indexed<Params>([&](std::size_t col){ return data[col * n + i]; });
 * @endcode
 */
template<class T, class Accessor> [[nodiscard]] constexpr T make_from_indexed(Accessor access)
{
    // Braced-init-list evaluation is left-to-right, so members are populated in
    // declaration order: T{ access(0), access(1), ..., access(arity-1) }.
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return T{access(Is)...};
    }(std::make_index_sequence<aggregate_arity<T>()>{});
}

/**
 * @brief Destructure a homogeneous-`double` aggregate into a `std::array` of its
 *        members, in declaration order.
 *
 * The inverse of make_from_indexed() for the all-`double` parameter structs the
 * library stores. Implemented with @c std::bit_cast, so @p T must be trivially
 * copyable and exactly `aggregate_arity<T>()` doubles wide (no padding) &mdash;
 * enforced by a @c static_assert.
 *
 * @tparam T A trivially copyable aggregate whose members are all `double`.
 * @return `std::array<double, aggregate_arity<T>()>` of the members in order.
 */
template<class T> [[nodiscard]] constexpr std::array<double, aggregate_arity<T>()> to_double_array(const T& value)
{
    constexpr std::size_t P = aggregate_arity<T>();
    static_assert(std::is_trivially_copyable_v<T>, "to_double_array<T>: T must be trivially copyable.");
    static_assert(sizeof(T) == P * sizeof(double),
                  "to_double_array<T>: T must be an all-`double` aggregate with no padding.");
    return std::bit_cast<std::array<double, P>>(value);
}

} // namespace glis::eos::detail
