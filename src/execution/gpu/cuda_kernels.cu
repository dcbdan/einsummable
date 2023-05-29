#include "cuda_kernels.h"


__global__ void touch1_none(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t0_offset_out, uint64_t t0_size, uint64_t t0_d_inn,uint64_t t0_d_out) {
    uint64_t index =  blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t inIndex = t0_offset_inn + index;
    uint64_t outIndex = t0_offset_out + index;

    if(index<t0_size){
        out[outIndex] = in[inIndex];    
    } 
}

__global__ void touch1_add(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t0_offset_out, uint64_t t0_size, uint64_t t0_d_inn,uint64_t t0_d_out) {
    uint64_t index =  blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t inIndex = t0_offset_inn + index;
    uint64_t outIndex = t0_offset_out + index;

    if(index<t0_size){
        out[outIndex] += in[inIndex];    
    } 
}

__global__ void touch1_mul(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t0_offset_out, uint64_t t0_size, uint64_t t0_d_inn,uint64_t t0_d_out) {
    uint64_t index =  blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t inIndex = t0_offset_inn + index;
    uint64_t outIndex = t0_offset_out + index;

    if(index<t0_size){
        out[outIndex] *= in[inIndex];    
    } 
}

__global__ void touch1_min(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t0_offset_out, uint64_t t0_size, uint64_t t0_d_inn,uint64_t t0_d_out) {
    uint64_t index =  blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t inIndex = t0_offset_inn + index;
    uint64_t outIndex = t0_offset_out + index;

    if(index<t0_size){
        out[outIndex] = fminf(out[outIndex],in[inIndex]);   
    } 
}

__global__ void touch1_max(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t0_offset_out, uint64_t t0_size, uint64_t t0_d_inn,uint64_t t0_d_out) {
    uint64_t index =  blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t inIndex = t0_offset_inn + index;
    uint64_t outIndex = t0_offset_out + index;

    if(index<t0_size){
        out[outIndex] = fmaxf(out[outIndex],in[inIndex]);   
    } 
}

__global__ void touch2_none(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t0_offset_out, uint64_t t1_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t1_d_inn,uint64_t t1_d_out) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
   
    uint64_t inRow = t0_offset_inn + row;
    uint64_t inCol = t1_offset_inn + col;
    uint64_t outRow = t0_offset_out + row;
    uint64_t outCol = t1_offset_out + col;

    if (row < t0_size && col < t1_size) {
        uint64_t inIndex = inRow * t1_d_inn + inCol;
        uint64_t outIndex = outRow * t1_d_out + outCol;
        out[outIndex] = in[inIndex];
    }
}

__global__ void touch2_add(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t0_offset_out, uint64_t t1_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t1_d_inn,uint64_t t1_d_out) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
   
    uint64_t inRow = t0_offset_inn + row;
    uint64_t inCol = t1_offset_inn + col;
    uint64_t outRow = t0_offset_out + row;
    uint64_t outCol = t1_offset_out + col;

    if (row < t0_size && col < t1_size) {
        uint64_t inIndex = inRow * t1_d_inn + inCol;
        uint64_t outIndex = outRow * t1_d_out + outCol;
        out[outIndex] += in[inIndex];
    }
}

__global__ void touch2_mul(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t0_offset_out, uint64_t t1_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t1_d_inn,uint64_t t1_d_out) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
   
    uint64_t inRow = t0_offset_inn + row;
    uint64_t inCol = t1_offset_inn + col;
    uint64_t outRow = t0_offset_out + row;
    uint64_t outCol = t1_offset_out + col;

    if (row < t0_size && col < t1_size) {
        uint64_t inIndex = inRow * t1_d_inn + inCol;
        uint64_t outIndex = outRow * t1_d_out + outCol;
        out[outIndex] *= in[inIndex];
    }
}

__global__ void touch2_min(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t0_offset_out, uint64_t t1_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t1_d_inn,uint64_t t1_d_out) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
   
    uint64_t inRow = t0_offset_inn + row;
    uint64_t inCol = t1_offset_inn + col;
    uint64_t outRow = t0_offset_out + row;
    uint64_t outCol = t1_offset_out + col;

    if (row < t0_size && col < t1_size) {
        uint64_t inIndex = inRow * t1_d_inn + inCol;
        uint64_t outIndex = outRow * t1_d_out + outCol;
        out[outIndex] = fminf(out[outIndex],in[inIndex]);
    }
}

__global__ void touch2_max(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t0_offset_out, uint64_t t1_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t1_d_inn,uint64_t t1_d_out) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
   
    uint64_t inRow = t0_offset_inn + row;
    uint64_t inCol = t1_offset_inn + col;
    uint64_t outRow = t0_offset_out + row;
    uint64_t outCol = t1_offset_out + col;

    if (row < t0_size && col < t1_size) {
        uint64_t inIndex = inRow * t1_d_inn + inCol;
        uint64_t outIndex = outRow * t1_d_out + outCol;
        out[outIndex] = fmaxf(out[outIndex],in[inIndex]);
    }
}


__global__ void touch3_none(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t2_size,uint64_t t1_d_inn,uint64_t t1_d_out,uint64_t t2_d_inn,uint64_t t2_d_out) {
    uint64_t xDim = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t yDim = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t zDim = blockIdx.z * blockDim.z + threadIdx.z;
    
   
    uint64_t inX = t0_offset_inn + xDim;
    uint64_t inY = t1_offset_inn + yDim;
    uint64_t inZ = t2_offset_inn + zDim;
    uint64_t outX = t0_offset_out + xDim;
    uint64_t outY = t1_offset_out + yDim;
    uint64_t outZ = t2_offset_out + zDim;

    if (xDim<t0_size&&yDim<t1_size&&zDim<t2_size) {
        uint64_t inIndex = inX * t1_d_inn *t2_d_inn+ inY*t2_d_inn+inZ;
        uint64_t outIndex = outX * t1_d_out *t2_d_out+ outY*t2_d_out+outZ;
        out[outIndex] = in[inIndex];
    }
}

__global__ void touch3_add(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t2_size,uint64_t t1_d_inn,uint64_t t1_d_out,uint64_t t2_d_inn,uint64_t t2_d_out) {
    uint64_t xDim = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t yDim = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t zDim = blockIdx.z * blockDim.z + threadIdx.z;
    
   
    uint64_t inX = t0_offset_inn + xDim;
    uint64_t inY = t1_offset_inn + yDim;
    uint64_t inZ = t2_offset_inn + zDim;
    uint64_t outX = t0_offset_out + xDim;
    uint64_t outY = t1_offset_out + yDim;
    uint64_t outZ = t2_offset_out + zDim;

    if (xDim<t0_size&&yDim<t1_size&&zDim<t2_size) {
        uint64_t inIndex = inX * t1_d_inn *t2_d_inn+ inY*t2_d_inn+inZ;
        uint64_t outIndex = outX * t1_d_out *t2_d_out+ outY*t2_d_out+outZ;
        out[outIndex] += in[inIndex];
    }
}

__global__ void touch3_mul(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t2_size,uint64_t t1_d_inn,uint64_t t1_d_out,uint64_t t2_d_inn,uint64_t t2_d_out) {
    uint64_t xDim = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t yDim = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t zDim = blockIdx.z * blockDim.z + threadIdx.z;
    
   
    uint64_t inX = t0_offset_inn + xDim;
    uint64_t inY = t1_offset_inn + yDim;
    uint64_t inZ = t2_offset_inn + zDim;
    uint64_t outX = t0_offset_out + xDim;
    uint64_t outY = t1_offset_out + yDim;
    uint64_t outZ = t2_offset_out + zDim;

    if (xDim<t0_size&&yDim<t1_size&&zDim<t2_size) {
        uint64_t inIndex = inX * t1_d_inn *t2_d_inn+ inY*t2_d_inn+inZ;
        uint64_t outIndex = outX * t1_d_out *t2_d_out+ outY*t2_d_out+outZ;
        out[outIndex] *= in[inIndex];
    }
}

__global__ void touch3_min(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t2_size,uint64_t t1_d_inn,uint64_t t1_d_out,uint64_t t2_d_inn,uint64_t t2_d_out) {
    uint64_t xDim = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t yDim = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t zDim = blockIdx.z * blockDim.z + threadIdx.z;
    
   
    uint64_t inX = t0_offset_inn + xDim;
    uint64_t inY = t1_offset_inn + yDim;
    uint64_t inZ = t2_offset_inn + zDim;
    uint64_t outX = t0_offset_out + xDim;
    uint64_t outY = t1_offset_out + yDim;
    uint64_t outZ = t2_offset_out + zDim;

    if (xDim<t0_size&&yDim<t1_size&&zDim<t2_size) {
        uint64_t inIndex = inX * t1_d_inn *t2_d_inn+ inY*t2_d_inn+inZ;
        uint64_t outIndex = outX * t1_d_out *t2_d_out+ outY*t2_d_out+outZ;
        out[outIndex] = fminf(out[outIndex],in[inIndex]);
    }
}

__global__ void touch3_max(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t2_size,uint64_t t1_d_inn,uint64_t t1_d_out,uint64_t t2_d_inn,uint64_t t2_d_out) {
    uint64_t xDim = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t yDim = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t zDim = blockIdx.z * blockDim.z + threadIdx.z;
    
   
    uint64_t inX = t0_offset_inn + xDim;
    uint64_t inY = t1_offset_inn + yDim;
    uint64_t inZ = t2_offset_inn + zDim;
    uint64_t outX = t0_offset_out + xDim;
    uint64_t outY = t1_offset_out + yDim;
    uint64_t outZ = t2_offset_out + zDim;

    if (xDim<t0_size&&yDim<t1_size&&zDim<t2_size) {
        uint64_t inIndex = inX * t1_d_inn *t2_d_inn+ inY*t2_d_inn+inZ;
        uint64_t outIndex = outX * t1_d_out *t2_d_out+ outY*t2_d_out+outZ;
        out[outIndex] = fmaxf(out[outIndex],in[inIndex]);
    }
}


void touch1_dispatch(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t0_offset_out, uint64_t t0_size, uint64_t t0_d_inn,uint64_t t0_d_out,cudaStream_t stream,uint64_t choice){
    dim3 blockSize(256); 
    dim3 gridSize((t0_size + blockSize.x - 1) / blockSize.x);

    if(choice==0){
        touch1_none<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t0_offset_out, t0_size,  t0_d_inn, t0_d_out);
    }
    else if(choice==1){
        touch1_add<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t0_offset_out, t0_size,  t0_d_inn, t0_d_out);
    }
    else if(choice==2){
        touch1_mul<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t0_offset_out, t0_size,  t0_d_inn, t0_d_out);
    }
    else if(choice==3){
        touch1_min<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t0_offset_out, t0_size,  t0_d_inn, t0_d_out);
    }
    else if(choice==4){
        touch1_max<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t0_offset_out, t0_size,  t0_d_inn, t0_d_out);
    }

}

void touch2_dispatch(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t0_offset_out, uint64_t t1_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t1_d_inn,uint64_t t1_d_out,cudaStream_t stream,uint64_t choice){
    dim3 blockSize(16, 16); 
    dim3 gridSize((t1_size + blockSize.x - 1) / blockSize.x, (t0_size + blockSize.y - 1) / blockSize.y);

    if(choice==0){
        touch2_none<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t0_offset_out, t1_offset_out, t0_size, t1_size, t1_d_inn, t1_d_out);
    }
    else if(choice==1){
        touch2_add<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t0_offset_out, t1_offset_out, t0_size, t1_size, t1_d_inn, t1_d_out);
    }
    else if(choice==2){
        touch2_mul<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t0_offset_out, t1_offset_out, t0_size, t1_size, t1_d_inn, t1_d_out);
    }
    else if(choice==3){
        touch2_min<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t0_offset_out, t1_offset_out, t0_size, t1_size, t1_d_inn, t1_d_out);
    }
    else if(choice==4){
        touch2_max<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t0_offset_out, t1_offset_out, t0_size, t1_size, t1_d_inn, t1_d_out);
    }

}

void touch3_dispatch(float* out, const float* in, uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,uint64_t t0_size,uint64_t t1_size,uint64_t t2_size,uint64_t t0_d_inn,uint64_t t0_d_out,uint64_t t1_d_inn,uint64_t t1_d_out,cudaStream_t stream,uint64_t choice){
    dim3 blockSize(8, 8, 8); 
    dim3 gridSize((t0_size + blockSize.x - 1) / blockSize.x, (t1_size + blockSize.y - 1) / blockSize.y, (t2_size + blockSize.z - 1) / blockSize.z);
    
    if(choice==0){
        touch3_none<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t2_offset_inn, t0_offset_out, t1_offset_out, t2_offset_out,t0_size, t1_size, t2_size,t1_d_inn, t1_d_out,t2_d_inn, t2_d_out);
    }
    else if(choice==1){
        touch3_add<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t2_offset_inn, t0_offset_out, t1_offset_out, t2_offset_out,t0_size, t1_size, t2_size,t1_d_inn, t1_d_out,t2_d_inn, t2_d_out);
    }
    else if(choice==2){
        touch3_mul<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t2_offset_inn, t0_offset_out, t1_offset_out, t2_offset_out,t0_size, t1_size, t2_size,t1_d_inn, t1_d_out,t2_d_inn, t2_d_out);
    }
    else if(choice==3){
        touch3_min<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t2_offset_inn, t0_offset_out, t1_offset_out, t2_offset_out,t0_size, t1_size, t2_size,t1_d_inn, t1_d_out,t2_d_inn, t2_d_out);
    }
    else if(choice==4){
        touch3_max<<<gridSize, blockSize,0,stream>>>(out, in, t0_offset_inn, t1_offset_inn, t2_offset_inn, t0_offset_out, t1_offset_out, t2_offset_out,t0_size, t1_size, t2_size,t1_d_inn, t1_d_out,t2_d_inn, t2_d_out);
    }

}