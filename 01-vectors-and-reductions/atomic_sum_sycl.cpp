#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <sycl/access/access.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/sycl.hpp>
#include <sycl/usm.hpp>

#include "benchmark.h"

constexpr size_t data_size = 32'768 * 8;
// constexpr size_t data_size = 196'908;
constexpr size_t local_size = 512;
constexpr size_t num_iter = 100;

using T = std::uint32_t;

int main() {
	T *data = static_cast<T *>(aligned_alloc(alignof(T), data_size * sizeof(T)));
	for (int i = 0; i < data_size; i++)
		data[i] = i % 16;

	// CPU sum for cross-checking
	T sum = 0;
	for (int i = 0; i < data_size; i++)
		sum += data[i];

	// GPU sum for performance
	sycl::queue q{};
	T *input = sycl::malloc_device<T>(data_size, q);
	T *gm_output = sycl::malloc_device<T>(1, q);
	T *lm_output = sycl::malloc_device<T>(1, q);

	std::vector<sycl::event> inputs_ready{
		q.copy(data, input, data_size),
		q.memset(gm_output, static_cast<T>(0), sizeof(T)),
		q.memset(lm_output, static_cast<T>(0), sizeof(T)),
	};

	auto global_atomic_kernel =
		[=](sycl::nd_item<1> item) {
			sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope_device, sycl::access::address_space::global_space>
				acc{*gm_output};

			acc.fetch_add(input[item.get_global_linear_id()]);
		};

	auto local_atomic_kernel = [=](sycl::local_accessor<T, 1> local_output) {
		return [=](sycl::nd_item<1> item) {
			sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope_work_group, sycl::access::address_space::local_space>
				local_acc{local_output[0]};
			sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope_device, sycl::access::address_space::global_space>
				global_acc{*lm_output};

			if (item.get_local_linear_id() == 0)
				local_acc.store(0);

			sycl::group_barrier(item.get_group());

			T partial = 0;

			for (int i = item.get_global_linear_id();
				 i < data_size;
				 i += item.get_global_range(0)) {
				partial += input[i];
			}

			local_acc.fetch_add(partial);

			sycl::group_barrier(item.get_group());

			if (item.get_local_linear_id() == 0)
				global_acc.fetch_add(local_acc.load());
		};
	};

	util::benchmark([&]() {
		sycl::event output_ready = q.submit([&](sycl::handler& h) {
			h.depends_on(inputs_ready);
			h.parallel_for(sycl::nd_range<1>{data_size, local_size}, global_atomic_kernel);
		});

		output_ready.wait_and_throw();
	},
					num_iter, "Reduce with global atomic refernece");
	q.memset(gm_output, static_cast<T>(0), sizeof(T)).wait_and_throw();

	util::benchmark([&]() {
		sycl::event output_ready = q.submit([&](sycl::handler& h) {
			h.depends_on(inputs_ready);
			sycl::local_accessor<T, 1> local_output{1, h};

			h.parallel_for(sycl::nd_range<1>{data_size / 8, local_size}, local_atomic_kernel(local_output));
		});

		output_ready.wait_and_throw();
	},
					num_iter, "Reduce with local atomic refernece");
	q.memset(lm_output, static_cast<T>(0), sizeof(T)).wait_and_throw();

	sycl::event goutput_ready = q.parallel_for(sycl::nd_range<1>{data_size, local_size}, inputs_ready, global_atomic_kernel);
	sycl::event loutput_ready = q.submit([&](sycl::handler& h) {
		h.depends_on(inputs_ready);
		sycl::local_accessor<T, 1> local_output{1, h};

		h.parallel_for(sycl::nd_range<1>{data_size / 8, local_size}, local_atomic_kernel(local_output));
	});

	T gdevice_sum = 0;
	T ldevice_sum = 0;

	q.copy(gm_output, &gdevice_sum, 1, goutput_ready);
	q.copy(lm_output, &ldevice_sum, 1, loutput_ready);

	q.wait_and_throw();

	std::cout << std::format("CPU: {}, global SYCL: {}, local SYCL: {}, differ: {}",
							 sum, gdevice_sum, ldevice_sum, sum != gdevice_sum && sum != ldevice_sum)
			  << std::endl;

	sycl::free(input, q);
	sycl::free(gm_output, q);
}
