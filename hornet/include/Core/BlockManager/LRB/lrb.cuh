#include "../../Conf/Common.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

////////////////////////////////////////////////////////////////
//LRB//
////////////////////////////////////////////////////////////////
template <typename degree_t>
constexpr int BitsPWrd = sizeof(degree_t)*8;

template <typename degree_t>
constexpr int NumberBins = sizeof(degree_t)*8 + 1;

template <typename degree_t>
__device__ inline
typename std::enable_if<(sizeof(degree_t) == 4), int>::type
ceilLog2_p1(degree_t val) {
  return BitsPWrd<degree_t> - __clz(val) + (__popc(val) > 1);
}

template <typename degree_t>
__device__ inline
typename std::enable_if<(sizeof(degree_t) == 8), int>::type
ceilLog2_p1(degree_t val) {
  return BitsPWrd<degree_t> - __clzll(val) + (__popcll(val) > 1);
}

template <typename degree_t>
__global__
void binDegrees(
    degree_t * bins,
    degree_t const * deg,
    degree_t count) {
  constexpr int BinCount = NumberBins<degree_t>;
  __shared__ degree_t lBin[BinCount];
  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) { lBin[i] = 0; }
  __syncthreads();

  for (degree_t i = threadIdx.x + (blockIdx.x*blockDim.x);
      i < count; i += gridDim.x*blockDim.x) {
    atomicAdd(lBin + ceilLog2_p1(deg[i]), 1);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    atomicAdd(bins + i, lBin[i]);
  }
}

template <typename vid_t, typename degree_t>
__global__
void rebinIds(
    vid_t * reorgV,
    vid_t const * v,
    degree_t * prefixBins,
    degree_t const * deg,
    degree_t count) {
  constexpr int BinCount = NumberBins<degree_t>;
  __shared__ degree_t lBin[BinCount];
  __shared__ int lPos[BinCount];
  if (threadIdx.x < BinCount) {
    lBin[threadIdx.x] = 0; lPos[threadIdx.x] = 0;
  }
  __syncthreads();

  degree_t tid = threadIdx.x + blockIdx.x*blockDim.x;
  int threadBin;
  degree_t threadPos;
  if (tid < count) {
    threadBin = ceilLog2_p1(deg[tid]);
    threadPos = atomicAdd(lBin + threadBin, 1);
  }
  __syncthreads();

  if (threadIdx.x < BinCount) {
    lPos[threadIdx.x] = atomicAdd(prefixBins + threadIdx.x, lBin[threadIdx.x]);
  }
  __syncthreads();

  if (tid < count) {
    reorgV[lPos[threadBin] + threadPos] = v[tid];
  }
}

template <typename vid_t, typename degree_t>
void
lrbDistribute(
    hornet::DArray<vid_t> &redistributedVertexIds,
    hornet::DArray<degree_t> &degreeDistributionBin,
    hornet::DArray<vid_t> &vertexIds,
    hornet::DArray<degree_t> &vertexDegrees,
    hornet::DArray<degree_t> &tempBin) {
  const unsigned BLOCK_SIZE = 512;
  unsigned blocks = (vertexIds.size() + BLOCK_SIZE - 1)/BLOCK_SIZE;
  binDegrees<degree_t><<<blocks, BLOCK_SIZE>>>(
      tempBin.data().get(),
      vertexDegrees.data().get(),
      vertexDegrees.size());
  thrust::exclusive_scan(tempBin.begin(), tempBin.end(), tempBin.begin());
  degreeDistributionBin = tempBin;
  rebinIds<int, int><<<blocks, BLOCK_SIZE>>>(
      redistributedVertexIds.data().get(), vertexIds.data().get(),
      tempBin.data().get(), vertexDegrees.data().get(), vertexIds.size());
}
