#pragma once

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <dpu>
#include <atomic>

#include "common.h"
#include "mram_allocator.hpp"

using namespace dpu;



namespace pim {

template<typename T>
using ShardedVector = std::vector<std::vector<T>>;

// Helper function for creating byte vectors
inline std::vector<std::uint8_t> bytes_from(const void *p, std::size_t n) {
  auto b = reinterpret_cast<const std::int8_t *>(p);
  return {b, b + n};
}

static std::atomic<bool> shutdown_mode{false};

struct ShutdownHandler {
    ~ShutdownHandler() {
        shutdown_mode.store(true);
    }
};
static ShutdownHandler shutdown_handler;

/**
 * @class PIMManager
 * @brief Singleton class managing DPU set, allocations, and data transfers
 * 
 * This class provides a centralized interface for managing DPU operations including:
 * - DPU initialization and configuration
 * - Memory allocation across multiple DPUs
 * - Data scatter/gather operations
 * - Kernel execution coordination
 */
class PIMManager {
private:
  mutable std::mutex pim_mutex_;

public:
  /**
   * @brief Initialize the PIM system with specified number of DPUs
   * @param nr_dpus Number of DPUs to allocate (default: 256)
   * @param elf Path to the DPU kernel binary (default: "main.dpu")
   */
  static void init(unsigned nr_dpus = DPU_ALLOCATE_ALL, std::string elf = "main.dpu") {

    PIMManager &m = instance();

    if (shutdown_mode.load()) {
      std::cerr << "PIMManager is in shutdown mode, cannot initialize." << std::endl;
      return;
    }

    std::lock_guard<std::mutex> lock(m.pim_mutex_);

    if (m.initialised_)
      return;
    m.sys_.reset(new DpuSet(DpuSet::allocate(nr_dpus)));
    nr_dpus = m.sys_->dpus().size();
    if (nr_dpus == 0) {
      throw std::runtime_error("No DPUs available for PIMManager initialization");
    }
    
    LOG_INFO("PIMManager => Initialized %u DPUs", nr_dpus);

    m.alloc_.reserve(nr_dpus);
    for (unsigned i = 0; i < nr_dpus; ++i)
      m.alloc_.push_back(std::make_unique<MramAllocator>());

    m.dpu_n_ = nr_dpus;
    m.elf_path_ = std::move(elf);
    m.initialised_ = true;
    m.ensure_loaded();
  }

  /**
   * @brief Get the singleton instance of PIMManager
   * @return Reference to the singleton instance
   */
  static PIMManager &instance() {
    static PIMManager m;
    return m;
  }

  /**
   * @brief Get the number of available DPUs
   * @return Number of DPUs
   */
  unsigned num_dpus() const { return dpu_n_; }

private:


public:
  /**
   * @brief Scatter different data to each DPU
   * @param per_dpu Vector of data buffers, one per DPU
   * @param dst_sym Destination symbol name
   * @param off Offset within the destination symbol
   */
  void scatter(std::vector<std::vector<dpu_word_t>> &per_dpu,
               uint32_t off = 0) {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    ensure_loaded();

    sys_->copy(DPU_MRAM_HEAP_POINTER_NAME, off, per_dpu);
  }

  /**
   * @brief Gather data from all DPUs
   * @param per Output vector to store gathered data
   * @param bytes Number of bytes to gather from each DPU
   * @param off Offset within the source symbol
   */
  void gather (ShardedVector<dpu_word_t> &data, size_t bytes,uint32_t off = 0) {

    std::lock_guard<std::mutex> lock(pim_mutex_);
    ensure_loaded();
    sys_->copy(data, bytes, DPU_MRAM_HEAP_POINTER_NAME, off);

    // return data;
  }

  /**
   * @brief Push arguments to DPU WRAM
   * @param p Pointer to argument data
   * @param sz Size of argument data
   * @param wram_sym WRAM symbol name for arguments
   */
  void push_args(const void *p, size_t sz,
                 const char *wram_sym = "DPU_INPUT_ARGUMENTS") {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    ensure_loaded();
    sys_->copy(wram_sym, 0, bytes_from(p, sz));
  }

  /**
   * @brief Memory block descriptor for DPU allocations
   */
  struct Block {
    uint32_t off;      ///< Memory offset
    size_t bytes;      ///< Block size in bytes
  };

  /**
   * @brief Allocate memory block on specified DPU
   * @param bytes Number of bytes to allocate
   * @return Block descriptor
   */
  Block allocate_uniform(size_t bytes) {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    bytes = align_up(bytes, 8);
    if (alloc_.empty()) throw std::runtime_error("allocators not ready");
    // allocate from first, then drain same amount
    uint32_t off = alloc_[0]->alloc(bytes);
    for (size_t i = 1; i < alloc_.size(); ++i)
      alloc_[i]->alloc(bytes);
    return {off, bytes};
  }

  /**
   * @brief Deallocate memory block
   * @param b Block descriptor to deallocate
   */
  void deallocate(const Block &b) { 
    if(shutdown_mode.load()) 
      return;
      
    std::lock_guard<std::mutex> lock(pim_mutex_);
    for (size_t i = 0; i < alloc_.size(); ++i)
        alloc_[i]->free(b.off, b.bytes); 
  }

  /**
   * @brief Execute kernel on all DPUs
   */
  void exec() {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    ensure_loaded();
    sys_->exec();
#ifdef DEBUG
    sys_->log(std::cout);
#endif
  }

  /**
   * @brief Get memory usage statistics for all DPUs
   * @return Vector of (allocated_bytes, total_capacity) pairs for each DPU
   */
  std::vector<std::pair<size_t, size_t>> get_memory_stats() const {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    std::vector<std::pair<size_t, size_t>> stats;
    stats.reserve(alloc_.size());
    for (const auto& allocator : alloc_) {
      stats.push_back(allocator->get_stats());
    }
    return stats;
  }

  /**
   * @brief Reset all DPU memory allocators
   */
  void reset_memory() {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    for (auto& allocator : alloc_) {
      allocator->reset();
    }
  }

private:
  PIMManager() = default;
  /**
   * @brief Ensure DPU kernel is loaded
   */
  void ensure_loaded() {
    if (!loaded_ && !elf_path_.empty()) {
      sys_->load(elf_path_);
      loaded_ = true;
    }
  }

  std::unique_ptr<DpuSet> sys_;                        ///< DPU system interface
  std::vector<std::unique_ptr<MramAllocator>> alloc_;  ///< Per-DPU memory allocators
  unsigned dpu_n_{0};                                  ///< Number of DPUs
  std::string elf_path_;                               ///< Path to kernel binary
  bool initialised_{false};                           ///< Initialization state
  bool loaded_{false};                                 ///< Kernel load state
};

} // namespace pim
