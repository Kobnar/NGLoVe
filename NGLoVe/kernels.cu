#include "kernels.cuh"

namespace GLoVe
{
	// Zeros an array
	__global__ void ZeroArrayKernel(
		const int length,
		float arr[]
	)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length)
			arr[i] = 0.0;
	}

	// Zeros an array
	__global__ void ZeroArrayKernel(
		const int length,
		double arr[]
	)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length)
			arr[i] = 0.0;
	}

	// Calcluates F_ij = f(X_ij)
	__global__ void CalcFKernel(
		const int C,
		const float X[],
		const float xmax, const float alpha,
		float F[]
	)
	{
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (c < C)
			if (X[c] < xmax)
				F[c] = pow(X[c] / xmax, alpha);
			else
				F[c] = 1.0;
	}

	// Calculates L_ij = log(X_ij + 1)
	__global__ void CalcLKernel(
		const int C,
		const float X[],
		float L[]
	)
	{
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (c < C)
			L[c] = log(X[c] + 1);
	}

	// Calculates M_ij = w_i.T @ w_j + b_i + b_j
	__global__ void CalcMKernel(
		const int C, const int P,
		const int rows[], const int cols[],
		const float W1[], const float W2[],
		const float b1[], const float b2[],
		float M[]
	)
	{
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (c < C)
		{
			int i = rows[c];
			int j = cols[c];

			M[c] = 0.0;
			for (int p = 0; p < P; p++)
				M[c] += W1[i * P + p] * W2[j * P + p];
			M[c] += b1[i] + b2[j];
		}
	}

	// Calculates FoML = row_sum( F_ij * (M_ij - L_ij) )
	__global__ void CalcFoMLKernel(
		const int C,
		const int rows[], const float F[], const float M[], const float L[],
		float FoML[]
	)
	{
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (c < C)
		{
			int k = rows[c];
			float val = F[c] * (M[c] - L[c]);
			atomicAdd(&FoML[k], val);
		}
	}

	// Calculates FoML = row_sum( F_ij * (M_ij - L_ij) )
	__global__ void CalcFoML2Kernel(
		const int C,
		const int rows[], const float F[], const float M[], const float L[],
		float FoML2[]
	)
	{
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (c < C)
		{
			int k = rows[c];
			float val = F[c] * pow(M[c] - L[c], 2);
			atomicAdd(&FoML2[k], val);
		}
	}

	// Calculates J = sum( F_ij * (M_ij - L_ij)^2 )
	__global__ void CalcJKernel(
		const int V,
		const float FoML2[],
		float* J
	)
	{
		int k = blockIdx.x * blockDim.x + threadIdx.x;

		if (k < V)
			atomicAdd(J, FoML2[k]);
	}

	// Calculates DWJ
	__global__ void CalcDWJKernel(
		const int V, const int P,
		const float W1[], const float W2[],
		const float FoML[],
		float DW1J[], float DW2J[],
		float GW1[], float GW2[]
	)
	{
		int k = blockIdx.x * blockDim.x + threadIdx.x;
		int p = threadIdx.y;

		if (k < V && p < P)
		{
			int kp = k * P + p;

			// Calculate derivatives
			float dw1j = W1[kp] * FoML[k];
			float dw2j = W2[kp] * FoML[k];

			// Update derivatives
			DW1J[kp] = dw1j;
			DW2J[kp] = dw2j;

			//// Update sums of squared gradients
			GW1[kp] += pow(dw1j, 2);
			GW2[kp] += pow(dw2j, 2);
		}
	}

	// Calculates DbJ
	__global__ void CalcDbJKernel(
		const int V,
		const float FoML[],
		float DbJ[],
		float Gb[]
	)
	{
		int k = blockIdx.x * blockDim.x + threadIdx.x;

		if (k < V)
		{
			// Calculate derivative
			//float db = 2 * FoML[k];
			float db = FoML[k];

			// Update derivative
			DbJ[k] = db;

			//// Update sum of squared gradients
			Gb[k] += pow(db, 2);
		}
	}
	__global__ void UpdateVectorKernel(
		const int V, const int P,
		const float eta,
		const float DW1J[], const float DW2J[],
		const float GW1[], const float GW2[],
		float W1[], float W2[]
	)
	{
		int k = blockIdx.x * blockDim.x + threadIdx.x;
		int p = threadIdx.y;

		if (k < V && p < P)
		{
			int kp = k * P + p;
			W1[kp] -= (eta / sqrt(GW1[kp] + 10e-8)) * DW1J[kp];
			W2[kp] -= (eta / sqrt(GW2[kp] + 10e-8)) * DW2J[kp];
		}
	}
	__global__ void UpdateBiasKernel(
		const int V,
		const float eta,
		const float DbJ[],
		const float Gb[],
		float b1[], float b2[]
	)
	{
		int k = blockIdx.x * blockDim.x + threadIdx.x;

		if (k < V)
		{
			b1[k] -= (eta / sqrt(Gb[k] + 10e-8)) * DbJ[k];
			b2[k] -= (eta / sqrt(Gb[k] + 10e-8)) * DbJ[k];
		}
	}
}