CXX := icpx
CXXFLAGS := -std=c++20 -fsycl
CPPFLAGS := -Ithird-party/sycl-academy/include

.PHONY: all clean

TARGET := build/atomic_sum

atomic_sum: ./01-vectors-and-reductions/atomic_sum_sycl.cpp
	${CXX} ${CXXFLAGS} ${CPPFLAGS} $< -o build/$@

reduce_sum: ./01-vectors-and-reductions/reduce_sycl.cpp
	${CXX} ${CXXFLAGS} ${CPPFLAGS} $< -o build/$@

build:
	mkdir -p $@

clean:
	rm -f build/*
