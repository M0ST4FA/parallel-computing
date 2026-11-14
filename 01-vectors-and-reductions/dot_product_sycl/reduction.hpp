#pragma once
#include "common.hpp"

struct ProductKernel {
	T *input;
	T *tmp;
	sycl::local_accessor<T, 1> scratchpad;

	void operator()(sycl::nd_item<1> item) const;
};
struct ProductKernel2 {
	T *input;
	T *tmp;

	void operator()(sycl::nd_item<1> item) const;
};
struct ReductionKernel {
	T *tmp;

	void operator()(sycl::nd_item<1> item, auto& contrib) const;
};
struct ReductionKernel2 {
	T *input;

	void operator()(sycl::nd_item<1> item, auto& contrib) const;
};

#include "reduction.tpp"
