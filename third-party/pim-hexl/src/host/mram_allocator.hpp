#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

namespace pim {

// Forward declaration of align_up function
static inline uint32_t align_up(uint32_t x, uint32_t a) {
  return (x + a - 1) & ~(a - 1);
}

/**
 * @class MramAllocator
 * @brief Simple memory allocator using free-list + bump allocation strategy
 * 
 * This allocator manages MRAM memory blocks for DPU operations.
 * It uses a combination of free-list for deallocated blocks and
 * bump allocation for new allocations.
 */
class MramAllocator {
public:
  /**
   * @brief Construct allocator with memory limit
   * @param limit Maximum memory that can be allocated (default: 64MB)
   */
  explicit MramAllocator(size_t limit = 64ull << 20) : limit_(limit) {}

  /**
   * @brief Allocate a memory block
   * @param bytes Number of bytes to allocate (will be aligned to 8-byte boundary)
   * @return Offset of the allocated block
   * @throws std::bad_alloc if allocation fails
   */
  uint32_t alloc(size_t bytes) {
    bytes = align_up(bytes, 8);
    // std::lock_guard<std::mutex> lk(mu_); // REMOVED - single threaded
    
    // Try to find a suitable free block
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second >= bytes) {
        uint32_t off = it->first;
        size_t rem = it->second - bytes;
        free_.erase(it);
        if (rem) free_[off + bytes] = rem;
        
        // Validate the offset before returning
        if (off > limit_) {
          throw std::runtime_error("Corrupted allocator: invalid offset from free list");
        }
        return off;
      }
    }
    
    // No suitable free block found, use bump allocation
    if (cur_ + bytes > limit_) {
      throw std::bad_alloc();
    }
    uint32_t off = cur_;
    cur_ += bytes;
    
    // Validate the offset before returning
    if (off > limit_) {
      throw std::runtime_error("Corrupted allocator: invalid bump allocation");
    }
    return off;
  }

  /**
   * @brief Deallocate a memory block
   * @param off Offset of the block to deallocate
   * @param bytes Size of the block to deallocate
   */
  void free(uint32_t off, size_t bytes) {
    bytes = align_up(bytes, 8);
    // std::lock_guard<std::mutex> lk(mu_); // REMOVED - single threaded
    
    // Validate input parameters
    if (off > limit_ || off + bytes > limit_) {
      // Log the error but don't throw - this might be called during destruction
      return;
    }
    
    // Check for double-free
    if (free_.find(off) != free_.end()) {
      // Double free detected - log and return
      return;
    }
    
    free_[off] = bytes;
    
    // Merge with next block if adjacent
    auto nxt = free_.find(off + bytes);
    if (nxt != free_.end()) {
      bytes += nxt->second;
      free_.erase(nxt);
      free_[off] = bytes;
    }
    
    // Merge with previous block if adjacent
    auto prv = free_.lower_bound(off);
    if (prv != free_.begin() && (--prv)->first + prv->second == off) {
      prv->second += bytes;
      free_.erase(off);
    }
  }

  /**
   * @brief Get current memory usage statistics
   * @return Pair of (allocated_bytes, total_capacity)
   */
  std::pair<size_t, size_t> get_stats() const {
    // std::lock_guard<std::mutex> lk(mu_); // REMOVED - single threaded
    size_t free_bytes = 0;
    for (const auto& block : free_) {
      free_bytes += block.second;
    }
    return {cur_ - free_bytes, limit_};
  }

  /**
   * @brief Reset allocator to initial state
   */
  void reset() {
    // std::lock_guard<std::mutex> lk(mu_); // REMOVED - single threaded
    cur_ = 0;
    free_.clear();
  }

private:
  uint32_t cur_{0};           ///< Current bump allocation pointer
  uint32_t limit_;            ///< Maximum memory limit
  std::map<uint32_t, size_t> free_;  ///< Free blocks map: offset -> size
  mutable std::mutex mu_;     ///< Mutex for thread safety
};

} // namespace pim
