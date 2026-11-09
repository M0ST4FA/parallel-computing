#include <format>
#include <sycl/accessor.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/sycl.hpp>

#include "benchmark.h"

constexpr size_t data_size = 32'768 * 32;
// constexpr size_t data_size = 196'908;
constexpr size_t local_size = 128;
constexpr size_t global_size = data_size / 32;
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

	try {

		// GPU sum for performance
		sycl::queue q{};
		T *input = sycl::malloc_device<T>(data_size, q);
		T *output = sycl::malloc_device<T>(1, q);

		std::vector<sycl::event> inputs_ready{
			q.copy(data, input, data_size),
			q.memset(output, static_cast<T>(0), sizeof(T)),
		};

		// clang-format off
		auto kernel =
		[&](sycl::handler& h) { 
			sycl::local_accessor<T, 1> scratch{local_size, h};

			return
			[=](sycl::nd_item<1> item) 
				{
					// 1. Reduce into private memory
					auto global_id = item.get_global_linear_id();
					auto global_size = item.get_global_range(0);

					auto local_id = item.get_local_linear_id();
					auto local_size = item.get_local_range(0);

					auto group = item.get_group();

					T partial = 0;
					for (size_t i = global_id; i < data_size; i += global_size)
						partial += input[i];

					scratch[local_id] = partial;

					sycl::group_barrier(item.get_group());

					// 2. Reduce locally
					for (size_t stride = scratch.size() / 2; stride > 0; stride /= 2) {
						if (local_id < stride)
							scratch[local_id] += scratch[local_id + stride];

						sycl::group_barrier(item.get_group());
					}

					// 3. Accumulate globally
					sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope_device, sycl::access::address_space::global_space>
						atomic_output{*output};

					if (group.leader())
						atomic_output.fetch_add(scratch[local_id]);
				}; 
			};
		// clang-format on

		// clang-format off
		auto group_algo_kernel =
		[&](sycl::handler& h) { 
			return
			[=](sycl::nd_item<1> item) 
				{
					// 1. Reduce into private memory
					auto global_id = item.get_global_linear_id();
					auto global_size = item.get_global_range(0);

					auto local_id = item.get_local_linear_id();
					auto local_size = item.get_local_range(0);

					T partial = 0;
					for (size_t i = global_id; i < data_size; i += global_size)
						partial += input[i];

					// 2. Reduce locally
					auto group = item.get_group();
					T group_sum = sycl::reduce_over_group(group, partial, sycl::plus<T>{});

					// 3. Accumulate globally
					sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope_device, sycl::access::address_space::global_space>
						atomic_output{*output};

					if (group.leader())
						atomic_output.fetch_add(group_sum);
				}; 
			};
		// clang-format on

		// clang-format off
		auto joing_reduce_kernel =
		[&](sycl::handler& h) { 
			return
			[=](sycl::nd_item<1> item) 
				{
					auto global_id = item.get_global_linear_id();
					auto global_size = item.get_global_range(0);

					auto local_id = item.get_local_linear_id();
					auto local_size = item.get_local_range(0);

					auto group_count = item.get_group_range(0);
					auto group_id = item.get_group_linear_id();
					size_t element_per_group = data_size / group_count;

					size_t begin = group_id * element_per_group;
					size_t end = (group_id + 1) * element_per_group;

					sycl::group<1> group = item.get_group(); 

					auto group_sum = sycl::joint_reduce(
							group,
							input + begin, 
							input + end, 
							sycl::plus<T>{});

					// 3. Accumulate globally
					sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope_device, sycl::access::address_space::global_space>
						atomic_output{*output};

					if (group.leader())
						atomic_output.fetch_add(group_sum);
				}; 
			};
		// clang-format on

		util::benchmark([&]() {
			sycl::event output_ready = q.submit([&](sycl::handler& h) {
				h.depends_on(inputs_ready);
				h.parallel_for(sycl::nd_range<1>{global_size, local_size}, kernel(h));
			});

			output_ready.wait_and_throw();
		},
						num_iter, "Reduce with (manual) tree reduction");

		q.memset(output, T(0), sizeof(T)).wait_and_throw();

		util::benchmark([&]() {
			sycl::event output_ready = q.submit([&](sycl::handler& h) {
				h.depends_on(inputs_ready);
				h.parallel_for(sycl::nd_range<1>{global_size, local_size}, group_algo_kernel(h));
			});

			output_ready.wait_and_throw();
		},
						num_iter, "Reduce with group algorithm");

		q.memset(output, T(0), sizeof(T)).wait_and_throw();

		util::benchmark([&]() {
			sycl::event output_ready = q.submit([&](sycl::handler& h) {
				h.depends_on(inputs_ready);
				h.parallel_for(sycl::nd_range<1>{global_size, local_size}, joing_reduce_kernel(h));
			});

			output_ready.wait_and_throw();
		},
						num_iter, "Reduce with joint_reduce");

		q.memset(output, T(0), sizeof(T)).wait_and_throw();

		sycl::event output_ready = q.submit([&](sycl::handler& h) {
			h.depends_on(inputs_ready);
			h.parallel_for(sycl::nd_range<1>{global_size, local_size}, joing_reduce_kernel(h));
		});

		T device_sum = 0;

		q.copy(output, &device_sum, 1, output_ready);

		q.wait_and_throw();

		std::cout << std::format("CPU: {}, SYCL: {}, differ: {}",
								 sum, device_sum, sum != device_sum)
				  << std::endl;

		sycl::free(input, q);
		sycl::free(output, q);
	} catch (sycl::exception e) {
		std::cout << e.what();
	}
}
