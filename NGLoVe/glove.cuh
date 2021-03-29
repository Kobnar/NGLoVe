#pragma once

#include <iostream>
#include <stdio.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.cuh"

namespace GLoVe
{
	cudaError_t train(
		const int C, const int V, const int P, const int T,
		const int rows[], const int cols[], const float X[],
		const float xmax, const float alpha, const float eta,
		float W1[], float W2[],
		float b1[], float b2[],
		float J[]
	);
}