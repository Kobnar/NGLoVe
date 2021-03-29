#pragma once

#include "pch.h"
#include <random>
#include <math.h>

namespace SGLoVe
{
	void initArrays(
		int V, int P,
		double W1[], double W2[],
		double b1[], double b2[],
		double GW1[],  double GW2[],
		double Gb[]
		);

	void CalcF(
		int C,
		double X[],
		double xmax, double alpha,
		double F[]
		);

	void CalcL(
		int C,
		double X[],
		double L[]
		);

	void CalcM(
		int C, int V, int P,
		int rows[], int cols[],
		double W1[], double W2[],
		double b1[], double b2[],
		double M[]
		);

	void CalcFoML(
		int C, int V,
		int rows[], double F[], double M[], double L[],
		double FoML[]
		);

	void CalcJ(
		int C,
		double F[], double M[], double L[],
		double *J
		);

	void CalcDJ(
		int V, int P,
		double W1[], double W2[],
		double b1[], double b2[],
		double FoML[],
		double DW1J[], double DW2J[],
		double DbJ[]
		);

	void AdaGrad(
		int V, int P,
		double W1[], double W2[],
		double b1[], double b2[],
		double DW1J[], double DW2J[],
		double DbJ[],
		double GW1[], double GW2[],
		double Gb[],
		double eta
		);

	void Train(
		int C, int V, int P,				// Dimensional parameters
		int rows[], int cols[], double X[],	// Sparse cooccurence matrix
		double xmax, double alpha,			// Scaling parameters
		double eta, int T,					// Training parameters
		double W[],							// Word vector matrix (output)
		double J[]							// Training error array (output)
	);

}