#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

namespace GLoVe
{
	__global__ void ZeroArrayKernel(
		const int length,
		float arr[]
	);

	__global__ void ZeroArrayKernel(
		const int length,
		double arr[]
	);

	__global__ void CalcFKernel(
		const int C,
		const float X[],
		const float xmax, const float alpha,
		float F[]
	);

	__global__ void CalcLKernel(
		const int C,
		const float X[],
		float L[]
	);

	__global__ void CalcMKernel(
		const int C, const int P,
		const int rows[], const int cols[],
		const float W1[], const float W2[],
		const float b1[], const float b2[],
		float M[]
	);

	__global__ void CalcFoMLKernel(
		const int C,
		const int rows[], const float F[], const float M[], const float L[],
		float FoML[]
	);

	__global__ void CalcFoML2Kernel(
		const int C,
		const int rows[], const float F[], const float M[], const float L[],
		float FoML2[]
	);

	__global__ void CalcJKernel(
		const int C,
		const float FoML2[],
		float* J
	);

	__global__ void CalcDWJKernel(
		const int V, const int P,
		const float W1[], const float W2[],
		const float FoML[],
		float DW1J[], float DW2J[],
		float GW1[], float GW2[]
	);

	__global__ void CalcDbJKernel(
		const int V,
		const float FoML[],
		float DbJ[],
		float Gb[]
	);

	__global__ void UpdateVectorKernel(
		const int V, const int P,
		const float eta,
		const float DW1J[], const float DW2J[],
		const float GW1[], const float GW2[],
		float W1[], float W2[]
	);

	__global__ void UpdateBiasKernel(
		const int V,
		const float eta,
		const float DbJ[],
		const float Gb[],
		float b1[], float b2[]
	);
}