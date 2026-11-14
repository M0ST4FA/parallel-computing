#pragma once
#include <sycl/sycl.hpp>

#include "joining.hpp"

void JoiningKernel::operator()(sycl::nd_item<1> item) const {
	const size_t element_per_group = vector_length / item.get_group_range(0);
	const size_t lid = item.get_local_id(0);
	const size_t local_stride = item.get_local_range(0);
	const size_t base = item.get_group(0) * element_per_group;

	sycl::group<1> group = item.get_group();

	for (size_t l = lid; l < element_per_group; l += local_stride)
		scratchpad[l] = 1;

	sycl::group_barrier(item.get_group());

	for (size_t l = lid; l < element_per_group; l += local_stride) {
		size_t g = base + l;

		scratchpad[l] = input[g] * input[g + vector_length];
	}

	sycl::group_barrier(group);

	auto group_sum = sycl::joint_reduce(group,
										&scratchpad[0],
										&scratchpad[0] + element_per_group,
										sycl::plus<T>{});

	sycl::atomic_ref<T, sycl::memory_order_relaxed,
					 sycl::memory_scope_device,
					 sycl::access::address_space::global_space>
		atomic_output{*output};

	if (group.leader())
		atomic_output.fetch_add(group_sum);
}
