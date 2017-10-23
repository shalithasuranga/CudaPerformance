#include <stdio.h>
#include <math.h>




const int TILE_WIDTH = 8;
__global__ void multi(float *a, float *b, float *c, int width) {
  __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float result = 0;
  
  for (int p = 0; p < width/TILE_WIDTH; p++) {
    s_a[threadIdx.y][threadIdx.x] = a[row*width + (p*TILE_WIDTH + threadIdx.x)];
    s_b[threadIdx.y][threadIdx.x] = b[(p*TILE_WIDTH + threadIdx.y)*width + col];


    __syncthreads();


    for (int i = 0; i < TILE_WIDTH; i++) {
      result += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
    }


    __syncthreads();
  }

  c[row * width + col] = result;
}



int main(int arg0, char **arg1) {
   cudaThreadSynchronize();
	
  int width = atoi(arg1[1]);
  int THREADS_PER_BLOCK = 64;
  if(arg0 == 3) THREADS_PER_BLOCK = atoi(arg1[2]);

  int sqrtThreads = sqrt(THREADS_PER_BLOCK);
  int nBlocks = width/sqrtThreads;
  if (width % sqrtThreads != 0) { 
    nBlocks++;
  }
  dim3 grid(nBlocks, nBlocks, 1);
  dim3 block(sqrtThreads, sqrtThreads, 1); 


  float *a_h;
  float *b_h;
  float *c_h; 
  float *d_h; 
  float *a_d;
  float *b_d;
  float *c_d;

  int size;


  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsed1;




  size = width * width * sizeof(float);
  
  a_h = (float*) malloc(size);
  b_h = (float*) malloc(size);
  c_h = (float*) malloc(size);
  d_h = (float*) malloc(size);


  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      a_h[i * width + j] = i;
      b_h[i * width + j] = i;
    }
  }


  cudaMalloc((void**)&a_d, size);
  cudaMalloc((void**)&b_d, size);
  cudaMalloc((void**)&c_d, size);


  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);


  multi<<<grid, block>>>(a_d, b_d, c_d, width);
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed1, start, stop);


  printf("%f\n", elapsed1/1000);


  cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
  
  

  free(a_h);
  free(b_h);
  free(c_h);
  free(d_h);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
