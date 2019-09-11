#include <cub/cub.cuh>
#include <thrust/device_vector.h>

constexpr unsigned WarpSize{32};
constexpr unsigned MULTIPLIER{4};
constexpr unsigned BitsPerWord{sizeof(unsigned) * 8};

template <typename degree_t>
__global__ void insertKernel(
    xlib::byte_t** edge_block_ptr,
    unsigned* vertex_offset,
    unsigned* edges_per_block,
    xlib::byte_t* blockPtr,
    unsigned blockSize,
    unsigned edgesPerBlock,
    unsigned * vacancyFlag,
    degree_t validityFlagCount,
    degree_t * validBlockIndex,
    degree_t requestedBlockCount,
    degree_t * currentBlockCount) {

  if (*currentBlockCount > requestedBlockCount) { return; }

  unsigned leaderId = {threadIdx.x / WarpSize};
  leaderId = leaderId * WarpSize;
  typedef cub::WarpScan<degree_t> WarpScan;
  __shared__ typename WarpScan::TempStorage tempStorage[MULTIPLIER];

  //Calculate number of flags this block is operating on.
  //Scan so that each thread is aware of how many valid flags exist before it in the block
  degree_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  degree_t threadPop = 0, totalBlockPop = 0, threadPopOffset = 0;
  unsigned data = 0;
  if (tid < validityFlagCount) { data = vacancyFlag[tid]; }
  threadPop = __popc(data);
  WarpScan(tempStorage[leaderId]).ExclusiveSum(threadPop, threadPopOffset, totalBlockPop);

  //Syncthreads so that shared memory can be reused

  //If there are no valid blocks for this thread block, exit function
  if (totalBlockPop == 0) { return; }

  //Get starting point of which flags this block will work on
  degree_t blockOffset = 0;
  if (threadIdx.x == leaderId) { blockOffset = atomicAdd(currentBlockCount, totalBlockPop); }
  blockOffset = __shfl_sync(0xffffffff, blockOffset, leaderId);

  //If this block is not required to return valid vacancyFlag indices, exit function
  if (blockOffset >= requestedBlockCount) { return; }

  degree_t globalThreadOffset = blockOffset + threadPopOffset;
  degree_t threadWriteItemCount = min(threadPop, requestedBlockCount - globalThreadOffset);
  degree_t iter = 0;
  while (iter < threadWriteItemCount) {
    degree_t shiftIndex = __ffs(data) - 1;
    data = data & ~(1<<shiftIndex);
    validBlockIndex[iter + globalThreadOffset] = tid*BitsPerWord + shiftIndex;
    iter++;
  }
  if (tid < validityFlagCount) { vacancyFlag[tid] = data; }
}

template <typename degree_t>
__global__ void insertKernel(
    unsigned * vacancyFlag,
    degree_t validityFlagCount,
    degree_t * validBlockIndex,
    degree_t requestedBlockCount,
    degree_t * currentBlockCount) {

  if (*currentBlockCount > requestedBlockCount) { return; }

  unsigned leaderId = {threadIdx.x / WarpSize};
  leaderId = leaderId * WarpSize;
  typedef cub::WarpScan<degree_t> WarpScan;
  __shared__ typename WarpScan::TempStorage tempStorage[MULTIPLIER];

  //Calculate number of flags this block is operating on.
  //Scan so that each thread is aware of how many valid flags exist before it in the block
  degree_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  degree_t threadPop = 0, totalBlockPop = 0, threadPopOffset = 0;
  unsigned data = 0;
  if (tid < validityFlagCount) { data = vacancyFlag[tid]; }
  threadPop = __popc(data);
  WarpScan(tempStorage[leaderId]).ExclusiveSum(threadPop, threadPopOffset, totalBlockPop);

  //Syncthreads so that shared memory can be reused

  //If there are no valid blocks for this thread block, exit function
  if (totalBlockPop == 0) { return; }

  //Get starting point of which flags this block will work on
  degree_t blockOffset = 0;
  if (threadIdx.x == leaderId) { blockOffset = atomicAdd(currentBlockCount, totalBlockPop); }
  blockOffset = __shfl_sync(0xffffffff, blockOffset, leaderId);

  //If this block is not required to return valid vacancyFlag indices, exit function
  if (blockOffset >= requestedBlockCount) { return; }

  degree_t globalThreadOffset = blockOffset + threadPopOffset;
  degree_t threadWriteItemCount = min(threadPop, requestedBlockCount - globalThreadOffset);
  degree_t iter = 0;
  while (iter < threadWriteItemCount) {
    degree_t shiftIndex = __ffs(data) - 1;
    data = data & ~(1<<shiftIndex);
    validBlockIndex[iter + globalThreadOffset] = tid*BitsPerWord + shiftIndex;
    iter++;
  }
  if (tid < validityFlagCount) { vacancyFlag[tid] = data; }
}

//vacancyFlag - flag depicting validity of blocks
//validBlockIndex - array to be filled with indices of valid blocks
template <typename degree_t>
degree_t insert(
    VertexAccess<degree_t>& ptr,
    xlib::byte_t* blockptr,
    unsigned blockSize,
    unsigned edgesPerBlock,
    unsigned * vacancyFlag,
    unsigned vacancyFlagSize,
    //degree_t * validBlockIndex,
    unsigned allocationOffset,
    unsigned allocationCount) {
  thrust::device_vector<degree_t> currentBlockCount(1, 0);
  int BlockSize = WarpSize*MULTIPLIER;
  insertKernel<degree_t><<<(vacancyFlagSize + BlockSize - 1)/BlockSize, BlockSize>>>(
      ptr.edge_block_ptr(), ptr.vertex_offset(), ptr.edges_per_block(),
      blockptr, blockSize, edgesPerBlock,
      vacancyFlag, vacancyFlagSize,
      allocationOffset, allocationCount,
      currentBlockCount.data().get());
  degree_t cb = currentBlockCount[0];
  if (cb >= allocationCount) { cb = allocationCount; }
  return cb;
}

//vacancyFlag - flag depicting validity of blocks
//validBlockIndex - array to be filled with indices of valid blocks
template <typename degree_t>
degree_t insert(
    VertexAccess<degree_t>& ptr,
    xlib::byte_t* blockptr,
    unsigned blockSize,
    unsigned edgesPerBlock,
    unsigned * vacancyFlag,
    degree_t vacancyFlagSize,
    degree_t * validBlockIndex,
    degree_t validBlockIndexSize) {
  thrust::device_vector<degree_t> currentBlockCount(1, 0);
  int BlockSize = WarpSize*MULTIPLIER;
  insertKernel<degree_t><<<(vacancyFlagSize + BlockSize - 1)/BlockSize, BlockSize>>>(
      vacancyFlag, vacancyFlagSize,
      validBlockIndex, validBlockIndexSize,
      currentBlockCount.data().get());
  degree_t cb = currentBlockCount[0];
  if (cb >= validBlockIndexSize) { cb = validBlockIndexSize; }
  return cb;
}

template <typename degree_t>
degree_t insert(
    VertexAccess<degree_t>& ptr,
    unsigned * vacancyFlag,
    degree_t vacancyFlagSize,
    degree_t * validBlockIndex,
    degree_t validBlockIndexSize) {
}

template<typename T, typename R>
HOST_DEVICE
R binarySearch(const T* mem, R size, T searched) {
    assert(size != 0 || std::is_signed<R>::value);
    R start = 0, end = size - 1;
    while (start <= end) {
        R mid = (start + end) / 2u;
        if (mem[mid] > searched)
            end = mid - 1;
        else if (mem[mid] < searched)
            start = mid + 1;
        else
            return mid;
    }
    return size; // indicate not found
}

template <typename degree_t>
__global__
void removeKernel(
    xlib::byte_t** edge_block_ptr,
    degree_t*       vertex_offset,
    degree_t    deallocationCount,
    xlib::byte_t ** blockPtr,
    unsigned ** vacancyPtr,
    degree_t blockCount) {
  degree_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > deallocationCount) { return; }

  xlib::byte_t* eptr = edge_block_ptr[tid];
  degree_t offset    = vertex_offset[tid];
  degree_t foundPos = binarySearch(blockPtr, blockCount, eptr);
  if (foundPos == blockCount) { return; }//Not found. This should not happen.

  degree_t* vacancyFlag = vacancyPtr[foundPos];
  atomicOr(vacancyFlag + (offset/BitsPerWord), 1<<(offset%BitsPerWord));
}

template <typename degree_t>
void
remove(VertexAccess<degree_t>& ptr,
    xlib::byte_t ** blockPtr,
    unsigned ** vacancyPtr,
    degree_t blockCount) {
  int BlockSize = WarpSize*MULTIPLIER;
  removeKernel<degree_t><<<(ptr.size() + BlockSize - 1)/BlockSize, BlockSize>>>(
      ptr.edge_block_ptr(), ptr.vertex_offset(), ptr.size(),
      blockPtr, vacancyPtr, blockCount);
}
