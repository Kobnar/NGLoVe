#include "pch.h"
#include "train.h"

namespace SGLoVe
{
	void initArrays(	
		int V, int P,
		double W1[], double W2[],
		double b1[], double b2[],
		double GW1[], double GW2[],
		double Gb[]
	)
	{
		std::default_random_engine rnd;
		for (int k = 0; k < V; k++)
		{
			for (int p = 0; p < P; p++)
			{
				W1[k * P + p] = rnd();
				W2[k * P + p] = rnd();

				GW1[k * P + p] = 0.0;
				GW2[k * P + p] = 0.0;
			}

			b1[k] = rnd();
			b2[k] = rnd();

			Gb[k] = 0.0;
		}
	}

	void CalcF(
		int C,
		double X[],
		double xmax, double alpha,
		double F[]
	)
	{
		for (int c = 0; c < C; c++)
			if (X[c] < xmax)
				F[c] = pow(X[c] / xmax, alpha);
			else
				F[c] = 1.0;
	}

	void CalcL(
		int C,
		double X[],
		double L[]
	)
	{
		for (int c = 0; c < C; c++)
			L[c] = log(X[c] + 1);
	}

	void CalcM(
		int C, int V, int P,
		int rows[], int cols[],
		double W1[], double W2[],
		double b1[], double b2[],
		double M[]
	)
	{
		for (int c = 0; c < C; c++)
		{
			int i = rows[c];
			int j = cols[c];
			M[c] = 0.0;
			for (int p = 0; p < P; p++)
				M[c] += W1[i * P + p] * W2[j * P + p];
			M[c] += b1[i] + b2[j];
		}
	}

	void CalcFoML(
		int C, int V,
		int rows[], double F[], double M[], double L[],
		double FoML[]
	)
	{
		for (int k = 0; k < V; k++)
			FoML[k] = 0.0;

		for (int c = 0; c < C; c++)
		{
			int k = rows[c];
			FoML[k] += F[c] * (M[c] - L[c]);
		}
	}

	void CalcJ(
		int C,
		double F[], double M[], double L[],
		double *J
	)
	{
		*J = 0.0;
		for (int c = 0; c < C; c++)
			*J += F[c] * pow(M[c] - L[c], 2);
	}

	void CalcDJ(
		int V, int P,
		double W1[], double W2[],
		double b1[], double b2[],
		double FoML[],
		double DW1J[], double DW2J[],
		double DbJ[]
	)
	{
		for (int k = 0; k < V; k++)
		{
			for (int p = 0; p < P; p++)
			{
				DW1J[k * P + p] = 2 * W1[k * P + p] * FoML[k];
				DW2J[k * P + p] = 2 * W2[k * P + p] * FoML[k];
			}
			DbJ[k] = 2 * FoML[k];
		}
	}

	void AdaGrad(
		int V, int P,
		double W1[], double W2[],
		double b1[], double b2[],
		double DW1J[], double DW2J[],
		double DbJ[],
		double GW1[], double GW2[],
		double Gb[],
		double eta
	)
	{
		for (int k = 0; k < V; k++)
		{
			// Calculate word vector matrix gradients and update values
			for (int p = 0; p < P; p++)
			{
				int kp = k * P + p;
				GW1[kp] += pow(DW1J[kp], 2);
				GW2[kp] += pow(DW2J[kp], 2);
				W1[kp] -= (eta / sqrt(GW1[kp] + 10e-8)) * DW1J[kp];
				W2[kp] -= (eta / sqrt(GW2[kp] + 10e-8)) * DW2J[kp];
			}

			// Calculate bias vector gradients and update values
			Gb[k] += pow(DbJ[k], 2);
			double b_delta = (eta / sqrt(Gb[k] + 10e-8)) * DbJ[k];
			b1[k] -= b_delta;
			b2[k] -= b_delta;
		}
	}

	void Train(
		int C, int V, int P,
		int rows[], int cols[], double X[],
		double xmax, double alpha,
		double eta, int T,
		double W[],
		double J[]
	)
	{
		// Allocate memory for sparse matrices w/ C non-zero elements
		double* F = new double[C];				// Sparse weight matrix
		double* L = new double[C];				// Sparse log(X + 1) matrix

		// Allocate memory for word vector matrices w/ VxP elements
		double* W1 = new double[V * P];			// Main word vector matrix
		double* W2 = new double[V * P];			// Context word vector matrix

		// Allocate memory for bias vectors w/ Vx1 elements
		double* b1 = new double[V];				// Main bias vector
		double* b2 = new double[V];				// Context bias vector

		// Allocate memory for gradient matrices w/ VxP elements
		double* DW1J = new double[V * P];		// Main word vector matrix derivative
		double* DW2J = new double[V * P];		// Context word vector matrix derivative

		// Allocate memory for gradient vectors w/ Vx1 elements
		double* DbJ = new double[V];			// Main/context bias vector derivative

		// Allocate memory for squared gradient sums w/ VxP elements
		double* GW1 = new double[V * P];		// Main word vector matrix sum of squared derivatives
		double* GW2 = new double[V * P];		// Context word vector matrix sum of squared derivatives

		// Allocate memory for squared gradient sums w/ VxP elements
		double* Gb = new double[V];				// Main/context bias vector sum of squared derivatives

		// Allocate memory for intermediate calculation arrays
		double* M = new double[C];				// W1 W2.T + b1 + b2.T
		double* FoML = new double[V];			// F * (M - L)

		initArrays(V, P, W1, W2, b1, b2, GW1, GW2, Gb);

		// Calculate weight and log-transformed sparse matrices
		CalcF(C, X, xmax, alpha, F);
		CalcL(C, X, L);

		// Iterate through the desired training steps
		for (int t = 0; t < T; t++)
		{
			CalcM(C, V, P, rows, cols, W1, W2, b1, b2, M);
			CalcFoML(C, V, rows, F, M, L, FoML);
			CalcJ(C, F, M, L, &J[t]);
			CalcDJ(V, P, W1, W2, b1, b2, FoML, DW1J, DW2J, DbJ);
			AdaGrad(V, P, W1, W2, b1, b2, DW1J, DW2J, DbJ, GW1, GW2, Gb, eta);
		}

		// Sum main and context word vector matrices
		for (int k = 0; k < V; k++)
			for (int p = 0; p < P; p++)
				W[k * P + p] = W1[k * P + p] + W2[k * P + p];

		// Release memory
		delete[] F;
		delete[] L;
		delete[] W1;
		delete[] W2;
		delete[] b1;
		delete[] b2;
		delete[] DW1J;
		delete[] DW2J;
		delete[] GW1;
		delete[] GW2;
		delete[] DbJ;
		delete[] Gb;
		delete[] M;
		delete[] FoML;
	}
}
