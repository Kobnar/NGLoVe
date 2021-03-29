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
		const double xmax, const double alpha, const double eta,
		double W1[], double W2[],
		double b1[], double b2[],
		double J[]
	);
}