
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
#define TILE_SIZE 16
#define CUDA_TIMING
#define DEBUG
#define HIST_SIZE 256
#define SCAN_SIZE HIST_SIZE*2

unsigned char *input_gpu;
unsigned char *output_gpu;
float *cdf;
unsigned int *hist_array;

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}
                
// Add GPU kernel and functions
// HERE!!!
__global__ void kernel_hist(unsigned char *input, 
                       unsigned int *hist, unsigned int height, unsigned int width){

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    int block_loc = threadIdx.y*TILE_SIZE+threadIdx.x;
    // HIST_SIZE 256
    __shared__ unsigned int hist_shared[HIST_SIZE];

    if (block_loc<HIST_SIZE) hist_shared[block_loc]=0;
    __syncthreads();

    if (x<width && y<height) atomicAdd(&(hist_shared[input[location]]),1);
    __syncthreads();

    if (block_loc<HIST_SIZE) {
    	atomicAdd(&(hist[block_loc]),hist_shared[block_loc]);
    }


}

__global__ void kernel_cdf(float *cdf, unsigned int *hist_array, int size){

	__shared__ float p[SCAN_SIZE];
	int tid=blockIdx.x*blockDim.x+threadIdx.x;

	if (tid<HIST_SIZE){
		p[tid]=hist_array[tid] / (float)size;
	}
	__syncthreads();

	for (int i=1; i<=HIST_SIZE;i*=2){
		int ai=(threadIdx.x+1)*i*2-1;
		if (ai<SCAN_SIZE) p[ai]+=p[ai-i];
		__syncthreads();
	}

	for (int i=HIST_SIZE/2;i>0;i/=2){
		int bi=(threadIdx.x+1)*i*2-1;
		if (bi+i<SCAN_SIZE) p[bi+i]+=p[bi];
		__syncthreads();
	}

	__syncthreads();
	if (tid<HIST_SIZE) cdf[tid]+=p[threadIdx.x];

}

__global__ void kernel_equlization(unsigned char *output, 
	                           unsigned char *input,
	                           float * cdf, unsigned int height, unsigned int width){
	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    float min=cdf[0]; float down=0.0F; float up=255.0F;

    if (x<width && y<height) {
    	float value=255.0F * (cdf[input[location]]-min) / (1.0F-min);
    	if (value<down) value=down;
    	if (value>up) value=up;
    	output[location]=(unsigned char) value; 

    }

}

void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
    
    float cdf_cpu[HIST_SIZE]={0};
    unsigned int hist_cpu[HIST_SIZE]={0};
	


	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&hist_array  , HIST_SIZE*sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&cdf         , size*sizeof(float)));
	// init output_gpu to 0
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
    checkCuda(cudaMemset(hist_array , 0 , HIST_SIZE*sizeof(unsigned int)));
    checkCuda(cudaMemset(cdf        , 0 , HIST_SIZE*sizeof(float)));
	
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimCdfGrid(1 + (size - 1) / HIST_SIZE);
    dim3 dimCdfBlock(HIST_SIZE);

	// Kernel Call
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        kernel_hist<<<dimGrid, dimBlock>>>(input_gpu, 
                                      hist_array,
                                      height,
                                      width);
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());

        kernel_cdf<<<dimCdfGrid,dimCdfBlock>>>(cdf, 
        	                          hist_array,
        	                          height*width);
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
   

// ///////////////////////////////////////////        
//         checkCuda(cudaMemcpy(hist_cpu,
//         	hist_array,
//         	HIST_SIZE*sizeof(unsigned int),
//         	cudaMemcpyDeviceToHost));
//         checkCuda(cudaDeviceSynchronize());
//         cdf_cpu[0]=hist_cpu[0]/ ((float) height*width);
//         for (int i=1;i<HIST_SIZE;i++){
//         	cdf_cpu[i]=cdf_cpu[i-1]+hist_cpu[i]/ ((float) height*width);
//         }
//         checkCuda(cudaMemcpy(cdf,
//         	cdf_cpu,
//         	HIST_SIZE*sizeof(float),
//         	cudaMemcpyHostToDevice));
//         checkCuda(cudaDeviceSynchronize());
// ///////////////////////////////////////////





        kernel_equlization<<<dimGrid, dimBlock>>>(output_gpu, 
        	                          input_gpu,
        	                          cdf, 
        	                          height,
        	                          width);

        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));

    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));

}

void histogram_gpu_warmup(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
                         
	printf("cold up \n");

}

