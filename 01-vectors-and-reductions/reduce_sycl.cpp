#include <format>
#include <sycl/accessor.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/sycl.hpp>

#include "benchmark.h"

constexpr size_t data_size = 32'768 * 8;
// constexpr size_t data_size = 196'908;
constexpr size_t local_size = 512;
constexpr size_t global_size = data_size;
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
					// 1. Load data into scratch space
					auto global_id = item.get_global_linear_id();
					auto global_size = item.get_global_range(0);

					auto local_id = item.get_local_linear_id();
					auto local_size = item.get_local_range(0);

					scratch[local_id] = input[global_id];

					sycl::group_barrier(item.get_group());

					// 2. Accumulate locally
					size_t rem = scratch.size();

					while (rem != 1) {
						if (local_id < rem / 2)
							scratch[local_id] += scratch[local_id + rem / 2];

						sycl::group_barrier(item.get_group());

						// Enter next stage; we now have half the number of items
						rem /= 2;
					}

					// 3. Accumulate globally
					sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope_device, sycl::access::address_space::global_space>
						atomic_output{*output};

					if (local_id == 0)
						atomic_output.fetch_add(scratch[local_id]);
				}; 
			};
		// clang-format on

		util::benchmark([&]() {
			sycl::event output_ready = q.submit([&](sycl::handler& h) {
				h.depends_on(inputs_ready);
				h.parallel_for(sycl::nd_range<1>{global_size, local_size}, kernel(h));
			});

			q.memset(output, T(0), sizeof(T), output_ready).wait_and_throw();
		},
						num_iter, "Reduce with tree reduction");

		sycl::event output_ready = q.submit([&](sycl::handler& h) {
			h.depends_on(inputs_ready);
			h.parallel_for(sycl::nd_range<1>{global_size, local_size}, kernel(h));
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
