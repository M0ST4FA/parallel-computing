#pragma once

#include <cassert>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <sycl/sycl.hpp>
#include <xmmintrin.h>

constexpr size_t vector_length = 32'768 * 32;
// constexpr size_t data_size = 196'908;
constexpr size_t local_size = 128;
constexpr size_t global_size = vector_length / 32;
constexpr size_t num_iter = 100;

using T = std::uint32_t;

typedef T v8u __attribute((vector_size(sizeof(T) * 8)));
typedef T v4u __attribute((vector_size(sizeof(T) * 4)));
