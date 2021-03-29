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
		const double xmax, const double alpha,
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
		const double W1[], const double W2[],
		const double b1[], const double b2[],
		float M[]
	);

	__global__ void CalcFoMLKernel(
		const int C,
		const int rows[], const float F[], const float M[], const float L[],
		double FoML[]
	);

	__global__ void CalcFoML2Kernel(
		const int C,
		const int rows[], const float F[], const float M[], const float L[],
		double FoML2[]
	);

	__global__ void CalcJKernel(
		const int C,
		const double FoML2[],
		double* J
	);

	__global__ void CalcDWJKernel(
		const int V, const int P,
		const double W1[], const double W2[],
		const double FoML[],
		double DW1J[], double DW2J[],
		double GW1[], double GW2[]
	);

	__global__ void CalcDbJKernel(
		const int V,
		const double FoML[],
		double DbJ[],
		double Gb[]
	);

	__global__ void UpdateVectorKernel(
		const int V, const int P,
		const double eta,
		const double DW1J[], const double DW2J[],
		const double GW1[], const double GW2[],
		double W1[], double W2[]
	);

	__global__ void UpdateBiasKernel(
		const int V,
		const double eta,
		const double DbJ[],
		const double Gb[],
		double b1[], double b2[]
	);
}