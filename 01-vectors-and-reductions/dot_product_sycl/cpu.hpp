#pragma once
#include "common.hpp"

using DotFunction = T (*)(const T *, const T *);

template <size_t step, class V, size_t... I>
V butterfly_stage(V value, std::index_sequence<I...>) {
	return value + __builtin_shufflevector(value, value, (I ^ step)...);
}

template <size_t step, class V>
V butterfly_reduce(V value) {
	if constexpr (step == 0)
		return value;

	constexpr size_t lanes = sizeof(V) / sizeof(T);
	V next = butterfly_stage<step>(value, std::make_index_sequence<lanes>{});

	return butterfly_reduce<step / 2>(next);
}

template <typename V>
T dot_portable(const T *ap, const T *bp) {
	V a, b;
	memcpy(&a, ap, sizeof(V));
	memcpy(&b, bp, sizeof(V));

	constexpr size_t vlen = sizeof(a) / sizeof(T);
	// static_assert(std::has_single_bit(vlen), "Vector length must be power of 2");
	static_assert((vlen & (vlen - 1)) == 0, "Vector length must be power of 2");

	V prod = a * b;
	V sum = butterfly_reduce<vlen / 2>(prod);

	return sum[0];
}

[[gnu::target("avx2")]]
T dot_avx2(const T *a, const T *b);

[[gnu::target("avx512f")]]
T dot_avx512(const T *ap, const T *bp);

DotFunction select_dot();
