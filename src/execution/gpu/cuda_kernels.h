#pragma once
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <stdint.h>
#include <chrono>
#include <math.h>

void touch1_dispatch(float*, const float*, uint64_t, uint64_t, uint64_t, uint64_t,uint64_t, cudaStream_t,uint64_t);

void touch2_dispatch(float*, const float*, uint64_t, uint64_t, uint64_t, uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,cudaStream_t,uint64_t);

void touch3_dispatch(float*, const float*, uint64_t, uint64_t, uint64_t, uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,cudaStream_t,uint64_t);