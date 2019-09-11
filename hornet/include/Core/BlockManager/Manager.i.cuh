namespace hornet {

#define BARR  BlockArray<TypeList<Ts...>, device_t, degree_t>
#define BCHN  BlockArrayChain<TypeList<Ts...>, device_t, degree_t>
#define BMNGR BlockArrayManager<TypeList<Ts...>, device_t, degree_t>

template<typename... Ts, DeviceType device_t, typename degree_t>
BARR::
BlockArray(const unsigned log2BlockCount, const degree_t log2Degree) noexcept :
  _edge_data(1<<(log2BlockCount+static_cast<unsigned>(log2Degree))),
  _vacancy_flags(1<<log2BlockCount, std::numeric_limits<unsigned>::max()),
  _vacancy_count(1<<log2BlockCount), _size(1<<log2BlockCount), _max_block_degree(1<<log2Degree) {
}

template<typename... Ts, DeviceType device_t, typename degree_t>
BARR::
BlockArray(const BARR& other) noexcept :
_edge_data(other._edge_data), _vacancy_flags(other._vacancy_flags),
  _vacancy_count(other._vacancy_count), _size(other._size), _max_block_degree(other._max_block_degree) {
}

template<typename... Ts, DeviceType device_t, typename degree_t>
BARR::
BlockArray(BARR&& other) noexcept :
_edge_data(std::move(other._edge_data)), _vacancy_flags(std::move(other._vacancy_flags)),
  _vacancy_count(other._vacancy_count), _size(other._size), _max_block_degree(other._max_block_degree) {
}


template<typename... Ts, DeviceType device_t, typename degree_t>
xlib::byte_t *
BARR::
get_blockarray_ptr(void) noexcept {
    return reinterpret_cast<xlib::byte_t *>(_edge_data.get_soa_ptr().template get<0>());
}

template<typename... Ts, DeviceType device_t, typename degree_t>
void
BARR::
insert(VertexAccess<degree_t>& ptr, unsigned* allocationCount, unsigned* allocationOffset) noexcept {
  if (_vacancy_count != _size) {
    //TODO : set all components of VertexAccess ptr correctly
    auto successfulAllocationCount =
      insert(ptr,
          get_blockarray_ptr(),
          static_cast<unsigned>(1<<log2Degree),
          capacity(),
          _vacancy_flags.data().get(), _vacancy_flags.size(),
          *allocationOffset, *allocationCount);
    (*allocationOffset) += successfulAllocationCount;
    (*allocationCount) -= successfulAllocationCount;
    _vacancy_count -= successfulAllocationCount;
  } else {
    insert_init(ptr, _vacancy_flags.data().get(), _vacancy_flags.size(), *allocationOffset, *allocationCount);
    (*allocationOffset) += (*allocationCount)
    _vacancy_count -= (*allocationCount)
    (*allocationCount) = 0;
  }
}

template<typename... Ts, DeviceType device_t, typename degree_t>
unsigned
BARR::
capacity(void) noexcept {
    return _edge_data.get_num_items();
}

template<typename... Ts, DeviceType device_t, typename degree_t>
size_t
BARR::
mem_size(void) noexcept {
    return xlib::SizeSum<Ts...>::value * capacity();
}

template<typename... Ts, DeviceType device_t, typename degree_t>
bool
BARR::
full(void) noexcept {
    return (_vacancy_count == 0);
}

template<typename... Ts, DeviceType device_t, typename degree_t>
CSoAData<TypeList<Ts...>, device_t>&
BARR::
get_soa_data(void) noexcept {
    return _edge_data;
}

//==============================================================================

template<typename... Ts, DeviceType device_t, typename degree_t>
BCHN::
BlockArrayChain(degree_t _log2BlockDegree) noexcept :
log2BlockDegree(_log2BlockDegree), maxLog2BlockArraySize(0) { }

template<typename... Ts, DeviceType device_t, typename degree_t>
void
BCHN::
insert(VertexAccess<degree_t>& ptr, unsigned allocationCount, unsigned allocationOffset,
    DArray<xlib::byte_t*>& blockPtr, DArray<unsigned*>& vacancyPtr) noexcept {
  if (allocationCount == 0) { return; }
  for (auto& block : blocks) {
    block.insert(ptr, &allocationCount, &allocationOffset);
    if (allocationCount == 0) break;
  }
  if (allocationCount != 0) {
    maxLog2BlockArraySize = std::max(xlib::ceil_log2(allocationCount), maxLog2BlockArraySize);
    blocks.emplace_front(maxLog2BlockArraySize, log2BlockDegree);
    auto &newBlock = blocks.front();
    newBlock.insert(ptr, &allocationCount, &allocationOffset);

    blockPtr.push_back(newBlock.get_blockarray_ptr());
    vacancyPtr.push_back(newBlock.get_vacancy_ptr());
  }
}

//==============================================================================

template<typename... Ts, DeviceType device_t, typename degree_t>
BMNGR::
BlockArrayManager(void) noexcept {
  for (int i = 0; i < sizeof(degree_t)*BITS_PER_WORD; ++i) { bchains.emplace_back(i); }
  lrbBin[0].resize(sizeof(degree_t)*BITS_PER_WORD + 2);
  lrbBin[1].resize(sizeof(degree_t)*BITS_PER_WORD + 2);
  blockPtr.reserve(8);
  vacancyPtr.reserve(8);
}

template<typename... Ts, DeviceType device_t, typename degree_t>
void
BMNGR::
remove(VertexAccess<degree_t>& ptr) noexcept {
  thrust::sort_by_key(blockPtr.begin(), blockPtr.end(), vacancyPtr.begin());
  remove(ptr, blockPtr, vacancyPtr, blockPtr.size());
}

template<typename... Ts, DeviceType device_t, typename degree_t>
void
BMNGR::
insert(DArray<vid_t> &vertexIds,
       DArray<degree_t> &vertexDegrees,
       DArray<vid_t> &relocVertexDegrees,
       VertexAccess<degree_t> &ptr) noexcept {
  //TODO : distribute vertexDegrees to ptr
  lrbDistribute(relocVertexDegrees, lrbBin[0],
      vertexIds, vertexDegrees, lrbBin[1]);
  HArray<unsigned> degreeDistributionOffset = lrbBin[0];
  //Vertices defined by the offset range degreeDistributionOffset[0] -> degreeDistributionOffset[1] are 0 degree vertices
  setNullPtr(ptr, degreeDistributionOffset[1] - degreeDistributionOffset[0], degreeDistributionOffset[0]);
  //TODO : degreeDistributionOffset length will differ based on signedness of degree_t
  for (unsigned i = 1; i < degreeDistributionOffset.size() - 1; ++i) {
    unsigned ceilLog2degree = i - 1;
    assert(ceilLog2degree < bchains.size());
    unsigned allocationCount = degreeDistributionOffset[i+1] - degreeDistributionOffset[i];
    if (allocationCount != 0) {
      bchains[ceilLog2degree].insert(
          ptr, allocationCount, degreeDistributionOffset[i], blockPtr, vacancyPtr);
    }
  }
}

}
