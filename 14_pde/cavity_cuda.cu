#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>


__global__ void calculateB(double* d_u, double* d_v, double* d_p, double* d_b,
                                  int nx, int ny, double dx, double dy, double dt, double rho, double nu) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int index = j * nx + i;

        d_b[index] = rho * (1 / dt *
                            ((d_u[index + 1] - d_u[index - 1]) / (2 * dx) + (d_v[index + nx] - d_v[index - nx]) / (2 * dy)) -
                            ((d_u[index + 1] - d_u[index - 1]) / (2 * dx)) * ((d_u[index + 1] - d_u[index - 1]) / (2 * dx)) -
                            2 * ((d_u[index + nx] - d_u[index - nx]) / (2 * dy)) * ((d_v[index + 1] - d_v[index - 1]) / (2 * dx)) -
                            ((d_v[index + nx] - d_v[index - nx]) / (2 * dy)) * ((d_v[index + nx] - d_v[index - nx]) / (2 * dy)));
    }
}

__global__ void calculateP(double* d_p, double* d_pn, double* d_b, double dx, double dy, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int index = j * nx + i;

        d_p[index] = (dy * dy * (d_pn[index + 1] + d_pn[index - 1]) +
            dx * dx * (d_pn[index + nx] + d_pn[index - nx]) -
            d_b[index] * dx * dx * dy * dy) /
            (2 * (dx * dx + dy * dy));
    }
}

__global__ void updateBoundaryP(double* d_p, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j < ny) {
        d_p[j * nx + (nx - 1)] = d_p[j * nx + (nx - 2)];
        d_p[j * nx] = d_p[j * nx + 1];
    }
    
    if (i < nx) {
        d_p[i] = d_p[nx + i];
        d_p[(ny - 1) * nx + i] = 0;
    }
 
}

__global__ void calculateUV(double* d_u, double* d_v, double* d_p, double* d_un, double* d_vn,
                         int nx, int ny, double dx, double dy, double dt, double rho, double nu) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int index = j * nx + i;

        d_u[index] = d_un[index] -
                     d_un[index] * dt / dx * (d_un[index] - d_un[index - 1]) -
                     d_un[index] * dt / dy * (d_un[index] - d_un[index - nx]) -
                     dt / (2 * rho * dx) * (d_p[index + 1] - d_p[index - 1]) +
                     nu * dt / (dx * dx) * (d_un[index + 1] - 2 * d_un[index] + d_un[index - 1]) +
                     nu * dt / (dy * dy) * (d_un[index + nx] - 2 * d_un[index] + d_un[index - nx]);

        d_v[index] = d_vn[index] -
                     d_vn[index] * dt / dx * (d_vn[index] - d_vn[index - 1]) -
                     d_vn[index] * dt / dy * (d_vn[index] - d_vn[index - nx]) -
                     dt / (2 * rho * dy) * (d_p[index + nx] - d_p[index - nx]) +
                     nu * dt / (dx * dx) * (d_vn[index + 1] - 2 * d_vn[index] + d_vn[index - 1]) +
                     nu * dt / (dy * dy) * (d_vn[index + nx] - 2 * d_vn[index] + d_vn[index - nx]);
    }
}

__global__ void updateBoundaryUV(double* d_u, double* d_v, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j < ny) {
        d_u[j * nx] = 0;
        d_u[j * nx + (nx - 1)] = 0;
        d_v[j * nx] = 0;
        d_v[j * nx + (nx - 1)] = 0;
    }
    
    if (i < nx) {
        d_u[i] = 0;
        d_u[(ny - 1) * nx + i] = 1;
        d_v[i] = 0;
        d_v[(ny - 1) * nx + i] = 0;
    }
 
}

int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0 / (nx - 1);
    double dy = 2.0 / (ny - 1);
    double dt = 0.01;
    double rho = 1.0;
    double nu = 0.02;

    std::vector<double> u(nx * ny, 0.0);
    std::vector<double> v(nx * ny, 0.0);
    std::vector<double> p(nx * ny, 0.0);
    std::vector<double> b(nx * ny, 0.0);

    std::vector<double> x(nx);
    std::vector<double> y(ny);

    // デバイスメモリの確保
    double* d_u;
    double* d_v;
    double* d_p;
    double* d_b;
    double* d_un;
    double* d_vn;
    double* d_pn;
    int* d_nx;
    int* d_ny;
    double* d_dx;
    double* d_dy;
    double* d_dt;
    double* d_rho;
    double* d_nu;    

    cudaMalloc((void**)&d_u, nx * ny * sizeof(double));
    cudaMalloc((void**)&d_v, nx * ny * sizeof(double));
    cudaMalloc((void**)&d_p, nx * ny * sizeof(double));
    cudaMalloc((void**)&d_b, nx * ny * sizeof(double));
    cudaMalloc((void**)&d_un, nx * ny * sizeof(double));
    cudaMalloc((void**)&d_vn, nx * ny * sizeof(double));
    cudaMalloc((void**)&d_pn, nx * ny * sizeof(double));
    cudaMalloc((void**)&d_nx, sizeof(int));
    cudaMalloc((void**)&d_ny, sizeof(int));
    cudaMalloc((void**)&d_dx, sizeof(double));
    cudaMalloc((void**)&d_dy, sizeof(double));
    cudaMalloc((void**)&d_dt, sizeof(double));
    cudaMalloc((void**)&d_rho, sizeof(double));
    cudaMalloc((void**)&d_nu, sizeof(double));

    for (int i = 0; i < nx; ++i)
        x[i] = i * dx;

    for (int i = 0; i < ny; ++i)
        y[i] = i * dy;

    // ホストからデバイスへのデータ転送
    //cudaMemcpy(d_u, u.data(), nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_v, v.data(), nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_p, p.data(), nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nx, &nx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ny, &ny, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, &dy, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, &dt, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, &rho, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridSize((nx + 15) / 16, (ny + 15) / 16);
    dim3 blockSize(16, 16);

    // 時間ステップのループ
    for (int n = 0; n < nt; ++n) {
        // ホストからデバイスへのデータ転送
        cudaMemcpy(d_u, u.data(), nx * ny * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v.data(), nx * ny * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p, p.data(), nx * ny * sizeof(double), cudaMemcpyHostToDevice);

        // 配列bを計算
        calculateB<<<gridSize, blockSize>>>(d_u, d_v, d_p, d_b, nx, ny, dx, dy, dt, rho, nu);

        // 圧力方程式を解く
        for (int it = 0; it < nit; ++it) {
            //cudaMemcpy(d_un, d_u, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
            //cudaMemcpy(d_vn, d_v, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_pn, d_p, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);

            //calculateB or 前のiterationのupdateBoundaryP の結果を待つ
            cudaDeviceSynchronize();

            calculateP<<<gridSize, blockSize>>>(d_p, d_pn, d_b, dx, dy, nx, ny);

            //calculatePの結果を待つ
            cudaDeviceSynchronize();
            updateBoundaryP<<<gridSize, blockSize>>>(d_p, nx, ny);
        }

        // uとvを計算
        cudaMemcpy(d_un, d_u, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vn, d_v, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);

        //updateBoundaryPの結果を待つ
        cudaDeviceSynchronize();
        cudaMemcpy(p.data(), d_p, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);

        calculateUV<<<gridSize, blockSize>>>(d_u, d_v, d_p, d_un, d_vn, nx, ny, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();

        cudaMemcpy(u.data(), d_u, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(v.data(), d_v, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);

        // 境界条件の設定
        for (int i = 0; i < nx; ++i) {
            u[i] = 0.0;
            u[(ny - 1) * nx + i] = 1.0;
            v[i] = 0.0;
            v[(ny - 1) * nx + i] = 0.0;
        }
        for (int j = 0; j < ny; ++j) {
            u[j * nx] = 0.0;
            u[j * nx + nx - 1] = 0.0;
            v[j * nx] = 0.0;
            v[j * nx + nx - 1] = 0.0;
        }

        // Output the results (optional)
        std::ofstream outFile("results_" + std::to_string(n) + ".txt");
        if (outFile.is_open()) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    outFile << x[i] << " " << y[j] << " " << p[j * nx + i] << " " << u[j * nx + i] << " " << v[j * nx + i] << "\n";
                }
            }   
            outFile.close();
        }
    }

    // デバイスメモリの解放
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);
    cudaFree(d_pn);
    cudaFree(d_nx);
    cudaFree(d_ny);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_dt);
    cudaFree(d_rho);
    cudaFree(d_nu);    

    return 0;
}
