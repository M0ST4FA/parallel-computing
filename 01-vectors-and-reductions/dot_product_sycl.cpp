#include <bit>
#include <cassert>
#include <cstring>
#include <emmintrin.h>
#include <format>
#include <immintrin.h>
#include <sycl/sycl.hpp>
#include <utility>
#include <xmmintrin.h>

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
T dot_avx2(const T *a, const T *b) {
	__m256i va = _mm256_loadu_si256((__m256i *)a);
	__m256i vb = _mm256_loadu_si256((__m256i *)b);
	constexpr size_t simd_width = sizeof(__m256i) / sizeof(T);
	__m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

	__m256i res = _mm256_mullo_epi32(va, vb);

	for (size_t step = simd_width / 2; step > 0; step /= 2) {
		__m256i indices = _mm256_xor_si256(lanes, _mm256_set1_epi32(step));

		__m256i shuf = _mm256_permutexvar_epi32(indices, res);
		res = _mm256_add_epi32(res, shuf);
	}

	// __m128i lo = _mm256_castsi256_si128(res);
	// __m128i hi = _mm256_extracti128_si256(res, 1);
	//
	// __m128i sum = _mm_add_epi32(lo, hi);
	//
	// __m128i shuf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1));
	// sum = _mm_add_epi32(sum, shuf);
	//
	// shuf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
	// sum = _mm_add_epi32(sum, shuf);

	return res[0];
}

[[gnu::target("avx512f")]]
T dot_avx512(const T *ap, const T *bp) {
	__m512i a = _mm512_loadu_si512(ap);
	__m512i b = _mm512_loadu_si512(bp);

	constexpr size_t simd_width = sizeof(__m512i) / sizeof(T);
	const __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

	__m512i res = _mm512_mullo_epi32(a, b);

	for (size_t step = simd_width / 2; step > 0; step /= 2) {
		__m512i indicies = _mm512_xor_si512(lanes, _mm512_set1_epi32(step));

		__m512 shuff = _mm512_permutexvar_epi32(indicies, res);
		res = _mm512_add_epi32(shuff, res);
	}

	return res[0];
}

using DotFunction = T (*)(const T *, const T *);

DotFunction select_dot() {
	if (__builtin_cpu_supports("avx512f")) {
		std::cout << "AVX512f" << std::endl;

		return dot_avx512;
	}

	if (__builtin_cpu_supports("avx2")) {
		std::cout << "AVX2" << std::endl;

		return dot_avx2;
	}

	return dot_portable<v8u>;
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
	DotFunction dot = select_dot();
	for (size_t i = 0; i < vector_length; i += lanes) {
		v8u a = *(v8u *)(vectors + i);
		v8u b;
		memcpy(&b, vectors + vector_length + i, sizeof(b));

		// Accumulate the contribution of the subvectors "a" and "b"
		dot_product += dot(vectors + i, vectors + vector_length + i);
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
