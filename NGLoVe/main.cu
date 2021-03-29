#include "main.cuh"

int train(
	const int C, const int V, const int P, const int T,
	const int rows[], const int cols[], const float X[],
	const float xmax, const float alpha, const float eta,
	float W1[], float W2[],
	float b1[], float b2[],
	float J[]
)
{
	cudaError_t cudaStatus = GLoVe::train(
		C, V, P, T,
		rows, cols, X,
		xmax, alpha, eta,
		W1, W2,
		b1, b2,
		J
	);

	return cudaStatus;
}