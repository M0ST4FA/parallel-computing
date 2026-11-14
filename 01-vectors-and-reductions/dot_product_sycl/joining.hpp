#pragma once
#include "common.hpp"

struct JoiningKernel {
	T *input;
	T *output;
	sycl::local_accessor<T, 1> scratchpad;

	void operator()(sycl::nd_item<1> item) const;
};

#include "joining.tpp"
