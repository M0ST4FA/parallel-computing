#pragma once
#include "common.hpp"

struct ManualKernel {
	T *input;
	T *output;
	sycl::local_accessor<T, 1> scratchpad;

	void operator()(sycl::nd_item<1> item) const;
};

#include "manual.tpp"
