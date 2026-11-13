#include "cpu.hpp"

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
