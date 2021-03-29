#include "glove.cuh"

namespace GLoVe
{
	cudaError_t train(
		const int C, const int V, const int P, const int T,
		const int rows[], const int cols[], const float X[],
		const double xmax, const double alpha, const double eta,
		double W1[], double W2[],
		double b1[], double b2[],
		double J[]
	)
	{
		// Init cudaError object
		cudaError_t cudaStatus;

		// Init logging hack (logs cout)
		freopen("cuda.log", "w", stdout);
		std::cout << "Initializing device...";

		// Choose which GPU to run on
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Device selection failed!" << std::endl;
			goto Error;
		}

		// Configure block/thread layouts

		dim3 lin_threads(1024);
		dim3 VP_threads(1024 / P, P);

		dim3 C_blocks(C / lin_threads.x + 1);
		dim3 V_blocks(V / lin_threads.x + 1);
		dim3 VP_blocks(V / VP_threads.x + 1);

		// Declare input array device memory pointers

		int* d_rows = 0;
		int* d_cols = 0;
		float* d_X = 0;

		// Declare output array device memory pointers

		double* d_W1 = 0;
		double* d_W2 = 0;

		double* d_b1 = 0;
		double* d_b2 = 0;

		double* d_J = 0;

		// Declare intermediate array device memory pointers

		float* d_F = 0;
		float* d_L = 0;
		float* d_M = 0;

		double* d_DW1J = 0;
		double* d_DW2J = 0;
		double* d_DbJ = 0;

		double* d_GW1 = 0;
		double* d_GW2 = 0;
		double* d_Gb = 0;

		double* d_FoML = 0;
		double* d_FoML2 = 0;

		// Allocate and copy cooccurrence matrix data

		cudaStatus = cudaMalloc((void**)&d_rows, C * sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_rows failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_cols, C * sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_cols failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_X, C * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_X failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_rows, rows, C * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying rows to d_rows failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_cols, cols, C * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying cols to d_cols failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_X, X, C * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying X to d_X failed!" << std::endl;
			goto Error;
		}

		// Scale and transform coccurrence matrix into f(X_ij) and log(X_ij + 1) arrays, then free up memory

		cudaStatus = cudaMalloc((void**)&d_F, C * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_F failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_L, C * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_L failed!" << std::endl;
			goto Error;
		}

		CalcFKernel <<< C_blocks, lin_threads >>> (C, d_X, xmax, alpha, d_F);
		CalcLKernel <<< C_blocks, lin_threads >>> (C, d_X, d_L);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			goto Error;

		cudaFree(d_X);


		// Allocate and copy output array memory onto device

		cudaStatus = cudaMalloc((void**)&d_W1, V * P * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_W1 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_W2, V * P * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_W2 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_b1, V * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_b1 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_b2, V * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_b2 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_J, T * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_J failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_W1, W1, V * P * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying W1 to d_W1 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_W2, W2, V * P * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying W2 to d_W2 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_b1, b1, V * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying b1 to d_b1 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_b2, b2, V * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying b2 to d_b2 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_J, J, T * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Coppying J to d_J failed!" << std::endl;
			goto Error;
		}

		// Allocate intermediate array memory on device

		cudaStatus = cudaMalloc((void**)&d_M, C * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_M failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_DW1J, V * P * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_DW1J failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_DW2J, V * P * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_DW2J failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_DbJ, V * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_DbJ failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_GW1, V * P * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_GW1 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_GW2, V * P * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_GW2 failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_Gb, V * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_Gb failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_FoML, V * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_FoML failed!" << std::endl;
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&d_FoML2, V * sizeof(double));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Allocating memory for d_FoML2 failed!" << std::endl;
			goto Error;
		}

		std::cout << "Done!\n" << std::endl;

		std::cout << "Using " << C_blocks.x << "x" << C_blocks.y << " blocks w/ " << lin_threads.x << "x" << lin_threads.y << " threads for " << C << "x1 sparse matrices." << std::endl;
		std::cout << "Using " << VP_blocks.x << "x" << VP_blocks.y << " blocks w/ " << VP_threads.x << "x" << VP_threads.y << " threads for " << V << "x" << P << " word vector matrices." << std::endl;
		std::cout << "Using " << V_blocks.x << "x" << V_blocks.y << " blocks w/ " << lin_threads.x << "x" << lin_threads.y << " threads for " << V << "x1 bias vectors." << std::endl;

		// Start overall timer

		std::cout << "\nTraining word vectors..." << std::endl;
		auto train_start = std::chrono::high_resolution_clock::now();

		// Zero initial gradient sums
		ZeroArrayKernel <<< V * P / lin_threads.x + 1, lin_threads >>> (V * P, d_GW1);
		ZeroArrayKernel <<< V * P / lin_threads.x + 1, lin_threads >>> (V * P, d_GW2);
		ZeroArrayKernel <<< V_blocks, lin_threads >>> (V, d_Gb);

		// Iterate through training epochs
		for (int t = 0; t < T; t++)
		{
			auto epoch_start = std::chrono::high_resolution_clock::now();
			std::cout << "\tEpoch #" << t;

			// Zero existing intermediate values
			ZeroArrayKernel <<< V_blocks, lin_threads >>> (V, d_FoML);
			ZeroArrayKernel <<< V_blocks, lin_threads >>> (V, d_FoML2);

			// Calculate w_i.T @ w_j + b_i + b_j
			CalcMKernel <<< C_blocks, lin_threads >>> (C, P, d_rows, d_cols, d_W1, d_W2, d_b1, d_b2, d_M);

			// Calculate F_ij * (M_ij - L_ij)
			CalcFoMLKernel <<< C_blocks, lin_threads >>> (C, d_rows, d_F, d_M, d_L, d_FoML);

			// Calculate F_ij * (M_ij - L_ij)^2
			CalcFoML2Kernel <<< C_blocks, lin_threads >>> (C, d_rows, d_F, d_M, d_L, d_FoML2);

			// Calculate error
			CalcJKernel <<< V_blocks, lin_threads >>> (V, d_FoML2, &d_J[t]);

			// Calculate Derivatives
			CalcDWJKernel <<< VP_blocks, VP_threads >>> (V, P, d_W1, d_W2, d_FoML, d_DW1J, d_DW2J, d_GW1, d_GW2);
			CalcDbJKernel <<< V_blocks, lin_threads >>> (V, d_FoML, d_DbJ, d_Gb);

			// Calculate batch descent step
			UpdateVectorKernel <<< VP_blocks, VP_threads >>> (V, P, eta, d_DW1J, d_DW2J, d_GW1, d_GW2, d_W1, d_W2);
			UpdateBiasKernel <<< V_blocks, lin_threads >>> (V, eta, d_DbJ, d_Gb, d_b1, d_b2);

			cudaDeviceSynchronize();

			auto epoch_end = std::chrono::high_resolution_clock::now();
			auto t_delta = epoch_end - epoch_start;
			std::cout << " (" << t_delta / std::chrono::seconds(1) << " s)" << std::endl;
		}

		auto train_end = std::chrono::high_resolution_clock::now();
		auto train_delta = train_end - train_start;
		std::cout << "\nTraining completed in " << train_delta / std::chrono::seconds(1) << " seconds." << std::endl;

		// Copy solutions back to host memory

		cudaStatus = cudaMemcpy(W1, d_W1, V * P * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			goto Error;

		cudaStatus = cudaMemcpy(W2, d_W2, V * P * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			goto Error;

		cudaStatus = cudaMemcpy(b1, d_b1, V * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			goto Error;

		cudaStatus = cudaMemcpy(b2, d_b2, V * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			goto Error;

		cudaStatus = cudaMemcpy(J, d_J, T * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			goto Error;

Error:

		// Free device memory

		cudaFree(d_rows);
		cudaFree(d_cols);

		cudaFree(d_W1);
		cudaFree(d_W2);

		cudaFree(d_b1);
		cudaFree(d_b2);

		cudaFree(d_J);

		cudaFree(d_F);
		cudaFree(d_L);
		cudaFree(d_M);

		cudaFree(d_DW1J);
		cudaFree(d_DW2J);
		cudaFree(d_DbJ);

		cudaFree(d_GW1);
		cudaFree(d_GW2);
		cudaFree(d_Gb);

		cudaFree(d_FoML);
		cudaFree(d_FoML2);

		// Reset device
		
		cudaDeviceReset();

		// Output error to log
		if (cudaStatus != cudaSuccess)
			std::cout << "!!! - ERROR DETECTED - !!!" << std::endl;

		return cudaStatus;
	}
}