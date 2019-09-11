#ifndef BLOCKARRAY_CUH
#define BLOCKARRAY_CUH
#include <thrust/device_vector.h>
#include <Host/Numeric.hpp>

namespace hornet {

template <typename Type>
using DArray = thrust::device_vector;

template <typename Type>
using HArray = thrust::host_vector;


template<typename... Ts, DeviceType device_t, typename degree_t>
class BlockArray<TypeList<Ts...>, device_t, degree_t> {

    template <typename, DeviceType, typename> friend class BlockArray;

    CSoAData<TypeList<Ts...>, device_t> _edge_data;
    DArray<unsigned>                _vacancy_flags;
    unsigned                        _vacancy_count;
    unsigned                                 _size;
    degree_t                     _max_block_degree;

    public:
    BlockArray(const unsigned log2BlockCount, const degree_t log2Degree) noexcept;

    BlockArray(const BlockArray<TypeList<Ts...>, device_t, degree_t>& other) noexcept;

    BlockArray(BlockArray<TypeList<Ts...>, device_t, degree_t>&& other) noexcept;

    ~BlockArray(void) noexcept = default;

    xlib::byte_t * get_blockarray_ptr(void) noexcept;

    degree_t *     get_vacancy_ptr(void) noexcept;

    degree_t insert(VertexAccess<degree_t>& ptr, degree_t allocationCount,   degree_t offset) noexcept;

    unsigned capacity(void) noexcept;

    size_t mem_size(void) noexcept;

    bool full(void) noexcept;

    CSoAData<TypeList<Ts...>, device_t>& get_soa_data(void) noexcept;
};

}

//Manage a chain of block arrays for a given ceil(log2(degree(vertex)))
template<typename... Ts, DeviceType device_t, typename degree_t>
class BlockArrayChain<TypeList<Ts...>, device_t, degree_t> {

  std::forward_list<BlockArray<TypeList<Ts...>, device_t, degree_t>> blocks;

  unsigned log2BlockDegree;
  unsigned maxLog2BlockArraySize;

  public:

  BlockArrayChain(unsigned _log2BlockDegree);

  void insert(VertexAccess<degree_t>& ptr, unsigned allocationCount, unsigned offset,
    DArray<xlib::byte_t*>& blockPtr, DArray<unsigned*>& vacancyPtr) noexcept;
}

template<typename... Ts, DeviceType device_t, typename degree_t>
class BlockArrayManager<TypeList<Ts...>, device_t, degree_t> {

  public:
  using CountType = std::make_unsigned<degree_t>::type;

  private:
  std::vector<BlockArrayChain<TypeList<Ts...>, device_t, unsigned> bchains;

  //maintain a map between blockPtrs and the associated vacancyPtr
  DArray<xlib::byte_t*> blockPtr;
  DArray<unsigned*>   vacancyPtr;
  DArray<unsigned> lrbBin[2];

  public:
  BlockArrayManager(void) noexcept;

  void insert(DArray<vid_t> &vertexIds,
              DArray<degree_t> &vertexDegrees,
              DArray<vid_t> &relocVertexDegrees,
              VertexAccess<degree_t> &ptr) noexcept;

  void remove(VertexAccess<degree_t>& ptr, degree_t deallocationCount) noexcept;
};

#include "Manager.i.cuh"
#endif
