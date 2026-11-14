#pragma once
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/sycl.hpp>

#include "manual.hpp"

void ManualKernel::operator()(sycl::nd_item<1> item) const {
	const size_t element_per_group = vector_length / item.get_group_range(0);
	const size_t lid = item.get_local_id(0);
	const size_t local_stride = item.get_local_range(0);
	const size_t base = item.get_group(0) * element_per_group;
	const size_t items_per_group = item.get_group_range(0);

	sycl::group<1> group = item.get_group();
	sycl::sub_group subgroup = item.get_sub_group();
	const size_t simd_width = subgroup.get_local_linear_range();

	for (size_t l = lid; l < element_per_group; l += local_stride)
		scratchpad[l] = 1;

	sycl::group_barrier(group);

	T sum = 0;
	for (size_t l = lid; l < element_per_group; l += local_stride) {
		size_t g = base + l;

		// scratchpad[l] = input[g] * input[g + vector_length];
		sum += input[g] * input[g + vector_length];
	}

	sycl::group_barrier(subgroup);

	for (size_t delta = simd_width / 2; delta > 0; delta /= 2) {
		T other = sycl::permute_group_by_xor(subgroup, sum, delta);

		sum += other;
		// No need for barrier as all subgroup members execute the loop in lockstep
	}

	sycl::atomic_ref<T, sycl::memory_order_relaxed,
					 sycl::memory_scope_device,
					 sycl::access::address_space::global_space>
		atomic_output{*output};

	if (subgroup.leader())
		atomic_output.fetch_add(sum);
}
