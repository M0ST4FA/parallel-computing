#include <bit>
#include <cassert>
#include <cstring>
#include <format>
#include <sycl/sycl.hpp>
#include <utility>

#include "benchmark.h"

constexpr size_t vector_length = 32'768 * 32;
// constexpr size_t data_size = 196'908;
constexpr size_t local_size = 128;
constexpr size_t global_size = vector_length / 32;
constexpr size_t num_iter = 100;

using T = std::uint32_t;

typedef T v8u __attribute((vector_size(sizeof(T) * 8)));
typedef T v4u __attribute((vector_size(sizeof(T) * 4)));

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
T calculate_dot_product(V a, V b) {
	constexpr size_t vlen = sizeof(a) / sizeof(T);
	// static_assert(std::has_single_bit(vlen), "Vector length must be power of 2");
	static_assert((vlen & (vlen - 1)) == 0, "Vector length must be power of 2");

	V prod = a * b;
	V sum = butterfly_reduce<vlen / 2>(prod);

	return sum[0];
}

int main() {
	// 2 vectors in one allocation for efficiency
	T *vectors = static_cast<T *>(aligned_alloc(alignof(T), vector_length * 2 * sizeof(T)));
	for (int i = 0; i < vector_length * 2; i++)
		vectors[i] = i % 16;

	// CPU sum for cross-checking
	// T dot_product = 0;
	// for (int i = 0; i < vector_length; i++)
	// 	dot_product += vectors[i] * vectors[vector_length + i];

	constexpr size_t lanes = sizeof(v8u) / sizeof(T);
	T dot_product = 0;
	for (size_t i = 0; i < vector_length; i += lanes) {
		v8u a = *(v8u *)(vectors + i);
		v8u b;
		memcpy(&b, vectors + vector_length + i, sizeof(b));

		// Accumulate the contribution of the subvectors "a" and "b"
		dot_product += calculate_dot_product(a, b);
	}

	try {

		// GPU sum for performance
		// sycl::queue q{};
		// T *input = sycl::malloc_device<T>(vector_length, q);
		// T *output = sycl::malloc_device<T>(1, q);
		//
		// std::vector<sycl::event> inputs_ready{
		// 	q.copy(vectors, input, vector_length),
		// 	q.memset(output, static_cast<T>(0), sizeof(T)),
		// };
		//
		T d_dot_product = 0;
		//
		// // q.copy(output, &d_dot_product, 1, output_ready);
		//
		// q.wait_and_throw();

		std::cout << std::format("CPU: {}, SYCL: {}, differ: {}",
								 dot_product, d_dot_product, dot_product != d_dot_product)
				  << std::endl;

		// sycl::free(input, q);
		// sycl::free(output, q);
	} catch (sycl::exception e) {
		std::cout << e.what();
	}
}
