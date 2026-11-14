#include <format>

#include "benchmark.h"

#include "common.hpp"

#include "cpu.hpp"

#include "joining.hpp"
#include "manual.hpp"
#include "reduction.hpp"

constexpr size_t num_iter = 100;

int main() {
	// 2 vectors in one allocation for efficiency
	T *vectors = static_cast<T *>(aligned_alloc(alignof(T), vector_length * 2 * sizeof(T)));
	for (int i = 0; i < vector_length * 2; i++)
		vectors[i] = i % 16;

	constexpr size_t simd_width = sizeof(v8u) / sizeof(T);
	T dot_product = 0;
	DotFunction dot = select_dot();
	for (size_t i = 0; i < vector_length; i += simd_width)
		// Accumulate the contribution of the subvectors "a" and "b"
		dot_product += dot(vectors + i, vectors + vector_length + i);

	try {

		sycl::queue q{};
		T *input = sycl::malloc_device<T>(vector_length * 2, q);
		T *tmp = sycl::malloc_device<T>(vector_length, q);
		T *output = sycl::malloc_device<T>(1, q);

		std::vector<sycl::event> inputs_ready{
			q.copy(vectors, input, vector_length),
			q.copy(vectors + vector_length, input + vector_length, vector_length),
			q.memset(output, static_cast<T>(0), sizeof(T)),
			// q.memset(tmp, static_cast<T>(1), sizeof(T)),
		};

		T d_dot_product = 0;

		util::benchmark([&]() {
			auto prod_finshed = q.submit([&](sycl::handler& h) {
				h.depends_on(inputs_ready);
				constexpr size_t group_number = global_size / local_size;
				constexpr size_t element_per_group = vector_length / group_number;
				sycl::local_accessor<T, 1> local_acc{element_per_group, h};

				h.parallel_for(sycl::nd_range<1>{global_size, local_size},
							   ProductKernel2(input, tmp));
			});

			auto reduction_finished = q.submit([&](sycl::handler& h) {
				h.depends_on(prod_finshed);
				auto reduction = sycl::reduction(output, sycl::plus<T>{});
				h.parallel_for(sycl::nd_range<1>{global_size, local_size},
							   reduction, ReductionKernel{tmp});
			});

			reduction_finished.wait_and_throw();
		},
						num_iter, "reduction with two kernels");

		q.memset(output, static_cast<T>(0), sizeof(T)).wait_and_throw();

		util::benchmark([&]() {
			auto reduction_finished = q.submit([&](sycl::handler& h) {
				h.depends_on(inputs_ready);
				auto reduction = sycl::reduction(output, sycl::plus<T>{});
				h.parallel_for(sycl::nd_range<1>{global_size, local_size},
							   reduction, ReductionKernel2{input});
			});

			reduction_finished.wait_and_throw();
		},
						num_iter, "reduction with one kernel");

		q.memset(output, static_cast<T>(0), sizeof(T)).wait_and_throw();

		util::benchmark([&]() {
			auto reduction_finished = q.submit([&](sycl::handler& h) {
				constexpr size_t group_number = global_size / local_size;
				constexpr size_t element_per_group = vector_length / group_number;
				sycl::local_accessor<T, 1> local_acc{element_per_group, h};

				h.depends_on(inputs_ready);
				h.parallel_for(sycl::nd_range<1>{global_size, local_size},
							   JoiningKernel{input, output, local_acc});
			});

			reduction_finished.wait_and_throw();
		},
						num_iter, "reduction with joining");

		q.memset(output, static_cast<T>(0), sizeof(T)).wait_and_throw();

		util::benchmark([&]() {
			auto reduction_finished = q.submit([&](sycl::handler& h) {
				constexpr size_t group_number = global_size / local_size;
				constexpr size_t element_per_group = vector_length / group_number;
				sycl::local_accessor<T, 1> local_acc{element_per_group, h};

				h.depends_on(inputs_ready);
				h.parallel_for(sycl::nd_range<1>{global_size, local_size},
							   ManualKernel{input, output, local_acc});
			});

			reduction_finished.wait_and_throw();
		},
						num_iter, "reduction with manual implementation");

		q.memset(output, static_cast<T>(0), sizeof(T)).wait_and_throw();

		auto reduction_finished = q.submit([&](sycl::handler& h) {
			constexpr size_t group_number = global_size / local_size;
			constexpr size_t element_per_group = vector_length / group_number;
			sycl::local_accessor<T, 1> local_acc{element_per_group, h};

			h.depends_on(inputs_ready);
			h.parallel_for(sycl::nd_range<1>{global_size, local_size},
						   ManualKernel{input, output, local_acc});
		});

		q.copy(output, &d_dot_product, 1, reduction_finished);

		q.wait_and_throw();

		std::cout << std::format("CPU: {}, SYCL: {}, differ: {}",
								 dot_product, d_dot_product, dot_product != d_dot_product)
				  << std::endl;

		// sycl::free(input, q);
		// sycl::free(output, q);
	} catch (sycl::exception e) {
		std::cout << e.what();
	}
}
