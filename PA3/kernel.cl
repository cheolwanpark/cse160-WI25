__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Compute C = A^T B 
  unsigned int i = get_global_id(0), j = get_global_id(1);
  if(i < numCRows && j < numCColumns) {
    unsigned int aidx=i, bidx=j, cidx = i*numCColumns+j;
    C[cidx] = 0;
    for(unsigned int k=0; k<numARows; ++k) {
      C[cidx] += A[aidx] * B[bidx];
      aidx += numAColumns;
      bidx += numBColumns;
    }
  }
}
