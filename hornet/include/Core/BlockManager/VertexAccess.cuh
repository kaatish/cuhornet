#include <thrust/device_vector.h>

namespace hornet {
namespace gpu {

template <typename degree_t>
class VertexAccess {

  DArray<degree_t>      _degree;
  DArray<xlib::byte_t*> _edge_block_ptr;
  DArray<degree_t>      _vertex_offset;
  DArray<degree_t>      _edges_per_block;

  public:
  VertexAccess(degree_t count = 0) noexcept;

  degree_t*      degree(void) noexcept;
  xlib::byte_t** edge_block_ptr(void) noexcept;
  degree_t*      vertex_offset(void) noexcept;
  degree_t*      edges_per_block(void) noexcept;

  degree_t       size(void) noexcept;
  void           resize(degree_t count) noexcept;
};

#define VERTEXACCESS VertexAccess<degree_t>

template <typename degree_t>
VERTEXACCESS::
VertexAccess(degree_t count) noexcept :
  _degree(count, 0),
  _edge_block_ptr(count, nullptr),
  _vertex_offset(count, 0),
  _edges_per_block(count, 0) {}

template <typename degree_t>
degree_t*
VERTEXACCESS::
degree(void) noexcept {
  return _degree.data().get();
}

template <typename degree_t>
xlib::byte_t**
VERTEXACCESS::
edge_block_ptr(void) noexcept {
  return _edge_block_ptr.data().get();
}

template <typename degree_t>
degree_t*
VERTEXACCESS::
vertex_offset(void) noexcept {
  return _vertex_offset.data().get();
}

template <typename degree_t>
degree_t*
VERTEXACCESS::
edges_per_block(void) noexcept {
  return _edges_per_block.data().get();
}

template <typename degree_t>
degree_t
VERTEXACCESS::
size(void) noexcept {
  return _degree.size();
}

template <typename degree_t>
void
VERTEXACCESS::
resize(degree_t count) noexcept {
  _degree.resize(count);
  _edge_block_ptr.resize(count);
  _vertex_offset.resize(count);
  _edges_per_block.resize(count);
}

} //namespace hornet
} //namespace gpu
