#pragma once

#include "common.hpp"
#include "reduction.hpp"

inline void ProductKernel::operator()(sycl::nd_item<1> item) const {
	const size_t element_per_group = vector_length / item.get_group_range(0);
	const size_t lid = item.get_local_id(0);
	const size_t local_stride = item.get_local_range(0);
	const size_t base = item.get_group(0) * element_per_group;

	for (size_t l = lid; l < element_per_group; l += local_stride)
		scratchpad[l] = 1;

	sycl::group_barrier(item.get_group());

	for (size_t l = lid; l < element_per_group; l += local_stride) {
		size_t g = base + l;

		scratchpad[l] = input[g] * input[g + vector_length];
	}

	sycl::group_barrier(item.get_group());

	for (size_t l = lid; l < element_per_group; l += local_stride)
		tmp[base + l] = scratchpad[l];
}
inline void ProductKernel2::operator()(sycl::nd_item<1> item) const {
	const size_t element_per_group = vector_length / item.get_group_range(0);
	const size_t element_per_item = element_per_group / item.get_local_range(0);
	const size_t simd_width = item.get_local_range(0);

	const size_t lid = item.get_local_linear_id();

	const size_t base = item.get_group(0) * element_per_group;

	for (size_t i = 0; i < element_per_item; i++) {
		size_t g = base + i * simd_width + lid;

		tmp[g] = input[g] * input[g + vector_length];
	}
}
void ReductionKernel::operator()(sycl::nd_item<1> item, auto& contrib) const {
	for (size_t i = item.get_global_id(0);
		 i < vector_length;
		 i += item.get_global_range(0))
		contrib.combine(tmp[i]);
}
