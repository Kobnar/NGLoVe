#include "main.cuh"

int train(
	const int C, const int V, const int P, const int T,
	const int rows[], const int cols[], const float X[],
	const double xmax, const double alpha, const double eta,
	double W1[], double W2[],
	double b1[], double b2[],
	double J[]
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