// ─────────────────────────── host.hpp ───────────────────────────
#pragma once
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dpu>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stddef.h>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "host_args.hpp"

using namespace dpu;

namespace pim {

// Global flag to track if we're in shutdown mode (to avoid segfaults during destruction)
static std::atomic<bool> shutdown_mode{false};

// Function to get a safe mutex that survives shutdown
static std::mutex& get_safe_console_mutex() {
  static std::mutex* safe_mutex = new std::mutex(); // Never destroyed
  return *safe_mutex;
}

// RAII helper to set shutdown mode
struct ShutdownHandler {
    ~ShutdownHandler() {
        shutdown_mode.store(true);
    }
};
static ShutdownHandler shutdown_handler;

static inline uint64_t align_up(uint64_t x, uint64_t a) {
  return (x + a - 1) & ~(a - 1);
}

inline std::vector<std::uint8_t> bytes_from(const void *p, std::size_t n) {
  auto b = reinterpret_cast<const std::int8_t *>(p);
  return {b, b + n};
}


/* MramAllocator */
class MramAllocator {
public:
  explicit MramAllocator(size_t limit = 64ull << 20) : limit_(limit) {}
  uint64_t alloc(size_t bytes) {
    bytes = align_up(bytes, 8);
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = free_.begin(); it != free_.end(); ++it)
      if (it->second >= bytes) {
        uint64_t off = it->first;
        size_t rem = it->second - bytes;
        free_.erase(it);
        if (rem)
          free_[off + bytes] = rem;
        return off;
      }
    if (cur_ + bytes > limit_)
      throw std::bad_alloc();
    uint64_t off = cur_;
    cur_ += bytes;
    return off;
  }
  void free(uint64_t off, size_t bytes) {
    bytes = align_up(bytes, 8);
    std::lock_guard<std::mutex> lk(mu_);
    free_[off] = bytes;
    auto nxt = free_.find(off + bytes);
    if (nxt != free_.end()) {
      bytes += nxt->second;
      free_.erase(nxt);
      free_[off] = bytes;
    }
    auto prv = free_.lower_bound(off);
    if (prv != free_.begin() && (--prv)->first + prv->second == off) {
      prv->second += bytes;
      free_.erase(off);
    }
  }

private:
  uint64_t cur_{0}, limit_;
  std::map<uint64_t, size_t> free_;
  std::mutex mu_;
};

/*  PIMManager  */
class PIMManager {
private:
  mutable std::mutex pim_mutex_;

public:
  static void init(unsigned nr_dpus = 8, std::string elf = "main.dpu") {
    PIMManager &m = instance();
    if (m.initialised_)
      return;
    m.sys_.reset(new DpuSet(DpuSet::allocate(nr_dpus)));
    nr_dpus = m.sys_->dpus().size();

    {
      std::lock_guard<std::mutex> lock(get_safe_console_mutex());
      std::cout << "PIMManager: Initializing with " << nr_dpus << std::endl;
    }

    m.alloc_.reserve(nr_dpus);
    for (unsigned i = 0; i < nr_dpus; ++i)
      m.alloc_.push_back(std::make_unique<MramAllocator>());

    m.dpu_n_ = nr_dpus;
    m.elf_path_ = std::move(elf);
    m.initialised_ = true;
    m.ensure_loaded();
  }

  static PIMManager &instance() {
    static PIMManager m;
    return m;
  }
  unsigned num_dpus() const { return dpu_n_; }

  /* bulk transfers */
  void broadcast(const void *src, size_t bytes, const char *dst_sym,
                 uint32_t off = 0) {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    broadcast_impl(src, bytes, dst_sym, off);
  }

private:
  void broadcast_impl(const void *src, size_t bytes, const char *dst_sym,
                      uint32_t off = 0) {
    ensure_loaded();
    sys_->copy(dst_sym, off, bytes_from(src, bytes), bytes);
  }

public:

  /* scatter: different buffer per DPU */
  void scatter(const std::vector<std::vector<uint64_t>> &per_dpu,
               const std::string &dst_sym = DPU_MRAM_HEAP_POINTER_NAME,
               uint32_t off = 0) {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    ensure_loaded();

    if (per_dpu.size() != dpu_n_)
      throw std::runtime_error("scatter size mismatch");

    sys_->copy(dst_sym, off, per_dpu);
  }

  void gather(std::vector<std::vector<uint64_t>> &per, size_t bytes,
              uint32_t off = 0) {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    ensure_loaded();
    per.assign(dpu_n_, std::vector<uint64_t>(bytes));
    sys_->copy(per, bytes, DPU_MRAM_HEAP_POINTER_NAME, off);
  }

  void push_args(const void *p, size_t sz,
                 const char *wram_sym = "DPU_INPUT_ARGUMENTS") {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    // Cast to dpu_arguments_t and print its content for debugging
    // const dpu_arguments_t *args = static_cast<const dpu_arguments_t *>(p);
    // debug_print_args(*args);
    // Proceed with actual broadcast (without additional locking)
    broadcast_impl(p, sz, wram_sym, 0);
  }

  /* MRAM allocation facade */
  struct Block {
    unsigned dpu;
    uint64_t off;
    size_t bytes;
  };

  Block allocate(unsigned id, size_t bytes) {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    uint64_t off = alloc_[id]->alloc(bytes);
    return {id, off, bytes};
  }

  void deallocate(const Block &b) { 
    std::lock_guard<std::mutex> lock(pim_mutex_);
    alloc_[b.dpu]->free(b.off, b.bytes); 
  }

  void exec() {
    std::lock_guard<std::mutex> lock(pim_mutex_);
    ensure_loaded();
    sys_->exec();
#ifdef DEBUG
    sys_->log(std::cout);
#endif
  }

private:
  PIMManager() = default;
  void ensure_loaded() {
    if (!loaded_ && !elf_path_.empty()) {
      sys_->load(elf_path_);
      loaded_ = true;
    }
  }

  std::unique_ptr<DpuSet> sys_;
  std::vector<std::unique_ptr<MramAllocator>> alloc_;
  std::atomic<unsigned> rr_{0};
  unsigned dpu_n_{0};

  std::string elf_path_;
  bool initialised_{false}, loaded_{false};
};

/* PIM Vector */
enum class CopyState : uint8_t { CLEAN, HOST_DIRTY, PIM_FRESH };

template <typename T> class Vector {
  struct Shard {
    PIMManager::Block blk;
    std::vector<T> host;
    mutable CopyState state{CopyState::HOST_DIRTY}; // dirty from start
  };

public:
  using Serializer = std::function<std::vector<uint64_t>(const std::vector<T>&)>;
  using Deserializer = std::function<void(const std::vector<uint64_t>&, std::vector<T>&)>;

  void set_serializer(Serializer s) { serializer_ = s; }
  void set_deserializer(Deserializer d) { deserializer_ = d; }

  // Default constructor
  Vector() = default;
  
  // Destructor - deallocate all blocks only when going out of scope
  ~Vector() {
    // During program shutdown, avoid calling PIMManager to prevent static destruction order issues
    // The DPU system will clean up the memory when it shuts down
    shards_.clear();
    total_ = 0;
  }
  
  // Copy constructor
  Vector(const Vector& other) : total_(0) {
    *this = other;
  }
  
  // Move constructor
  Vector(Vector&& other) noexcept {
    *this = std::move(other);
  }
  
  // Copy assignment
  Vector& operator=(const Vector& other) {
    if (this != &other) {
      // Simple approach: rebuild to match other's size, then copy data
      serializer_ = other.serializer_;
      deserializer_ = other.deserializer_;
      
      if (other.total_ > 0) {
        // Resize to match other and copy element by element
        build(other.total_, T{});
        for (size_t i = 0; i < other.total_; ++i) {
          (*this)[i] = other[i];
        }
      } else {
        // Other is empty
        build(0, T{});
      }
    }
    return *this;
  }
  
  // Move assignment
  Vector& operator=(Vector&& other) noexcept {
    if (this != &other) {
      // Don't use swap - instead move data and leave other empty
      // This avoids double-destruction issues
      
      // Clear our current data (this will deallocate in destructor)
      shards_.clear();
      total_ = 0;
      
      // Take ownership of other's data
      total_ = other.total_;
      serializer_ = std::move(other.serializer_);
      deserializer_ = std::move(other.deserializer_);
      shards_ = std::move(other.shards_);
      
      // Leave other in a valid empty state
      other.total_ = 0;
      other.serializer_ = nullptr;
      other.deserializer_ = nullptr;
      // other.shards_ is already empty after the move
    }
    return *this;
  }
  
  explicit Vector(size_t n) { build(n, T{}); }
  Vector(size_t n, const T &v) { build(n, v); }
  
  // STL-like interface
  size_t size() const { return total_; }
  bool empty() const { return total_ == 0; }
  
  // Clear all data but keep allocated blocks (deallocation only happens in destructor)
  void clear() {
    build(0, T{});
  }
  
  // Resize method
  void resize(size_t new_size) {
    if (new_size == total_) return;
    
    // Use build method which now handles reuse of existing blocks
    build(new_size, T{});
  }
  
  void resize(size_t new_size, const T& value) {
    if (new_size == total_) return;
    
    // Use build method which now handles reuse of existing blocks
    build(new_size, value);
  }
  
  // Iterator support
  class iterator {
    Vector* buffer_;
    size_t index_;
    
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    
    iterator(Vector* buffer, size_t index) : buffer_(buffer), index_(index) {}
    
    reference operator*() { return (*buffer_)[index_]; }
    pointer operator->() { return &((*buffer_)[index_]); }
    
    iterator& operator++() { ++index_; return *this; }
    iterator operator++(int) { iterator tmp = *this; ++index_; return tmp; }
    iterator& operator--() { --index_; return *this; }
    iterator operator--(int) { iterator tmp = *this; --index_; return tmp; }
    
    iterator& operator+=(difference_type n) { index_ += n; return *this; }
    iterator& operator-=(difference_type n) { index_ -= n; return *this; }
    iterator operator+(difference_type n) const { return iterator(buffer_, index_ + n); }
    iterator operator-(difference_type n) const { return iterator(buffer_, index_ - n); }
    
    difference_type operator-(const iterator& other) const { return index_ - other.index_; }
    
    bool operator==(const iterator& other) const { return index_ == other.index_; }
    bool operator!=(const iterator& other) const { return index_ != other.index_; }
    bool operator<(const iterator& other) const { return index_ < other.index_; }
    bool operator<=(const iterator& other) const { return index_ <= other.index_; }
    bool operator>(const iterator& other) const { return index_ > other.index_; }
    bool operator>=(const iterator& other) const { return index_ >= other.index_; }
  };
  
  class const_iterator {
    const Vector* buffer_;
    size_t index_;
    
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;
    
    const_iterator(const Vector* buffer, size_t index) : buffer_(buffer), index_(index) {}
    
    reference operator*() const { return (*buffer_)[index_]; }
    pointer operator->() const { return &((*buffer_)[index_]); }
    
    const_iterator& operator++() { ++index_; return *this; }
    const_iterator operator++(int) { const_iterator tmp = *this; ++index_; return tmp; }
    const_iterator& operator--() { --index_; return *this; }
    const_iterator operator--(int) { const_iterator tmp = *this; --index_; return tmp; }
    
    const_iterator& operator+=(difference_type n) { index_ += n; return *this; }
    const_iterator& operator-=(difference_type n) { index_ -= n; return *this; }
    const_iterator operator+(difference_type n) const { return const_iterator(buffer_, index_ + n); }
    const_iterator operator-(difference_type n) const { return const_iterator(buffer_, index_ - n); }
    
    difference_type operator-(const const_iterator& other) const { return index_ - other.index_; }
    
    bool operator==(const const_iterator& other) const { return index_ == other.index_; }
    bool operator!=(const const_iterator& other) const { return index_ != other.index_; }
    bool operator<(const const_iterator& other) const { return index_ < other.index_; }
    bool operator<=(const const_iterator& other) const { return index_ <= other.index_; }
    bool operator>(const const_iterator& other) const { return index_ > other.index_; }
    bool operator>=(const const_iterator& other) const { return index_ >= other.index_; }
  };
  
  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, total_); }
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, total_); }
  const_iterator cbegin() const { return const_iterator(this, 0); }
  const_iterator cend() const { return const_iterator(this, total_); }
  
  // Element access
  T& at(size_t index) {
    if (index >= total_) throw std::out_of_range("Vector index out of range");
    return (*this)[index];
  }
  
  const T& at(size_t index) const {
    if (index >= total_) throw std::out_of_range("Vector index out of range");
    return (*this)[index];
  }

  // Cereal serialization support
  template <class Archive>
  void save(Archive& ar) const {
    pull_all(); // Ensure all data is on host
    ar(total_);
    
    // Serialize all elements sequentially
    for (size_t i = 0; i < total_; ++i) {
      ar((*this)[i]);
    }
  }
  
  template <class Archive>
  void load(Archive& ar) {
    size_t new_size;
    ar(new_size);
    
    // Resize to accommodate the data
    if (new_size != total_) {
      resize(new_size);
    }
    
    // Load all elements sequentially
    for (size_t i = 0; i < new_size; ++i) {
      T value;
      ar(value);
      (*this)[i] = value;
    }
  }
  
  /* host access */
  T &operator[](size_t i) {
    pull_all();
    std::pair<size_t, size_t> loc = locate(i);
    size_t s = loc.first;
    size_t idx = loc.second;
    shards_[s].state = CopyState::HOST_DIRTY;
    return shards_[s].host[idx];
  }

  const T &operator[](size_t i) const {
    pull_all(); // bulk sync if needed
    std::pair<size_t, size_t> loc = locate(i);
    size_t s = loc.first;
    size_t idx = loc.second;
    return shards_[s].host[idx];
  }

  const std::vector<Shard> &shards() const { return shards_; }

  /* DMA helpers */
  void commit() const {
    auto &mgr = PIMManager::instance();
    // bucket by (offset,bytes)
    // std::map<std::pair<uint64_t, size_t>, std::vector<std::vector<uint64_t>>>
    //     buckets;

    // for (auto &sh : shards_) {
    //   if (sh.state != CopyState::HOST_DIRTY)
    //     continue;
    //   std::pair<uint64_t, size_t> key(sh.blk.off, sh.blk.bytes);
    //   auto &per = buckets[key];
    //   if (per.empty())
    //     per.resize(mgr.num_dpus());
      
    //   // Use custom serializer if available, otherwise use raw bytes
    //   if (serializer_) {
    //     per[sh.blk.dpu] = serializer_(sh.host);
    //   } 
      
    //   // else {
    //   //   per[sh.blk.dpu] = sh.host;
    //   // }
    //   sh.state = CopyState::CLEAN;
    // }

    unsigned num_dpus = mgr.num_dpus();
    std::vector<std::vector<uint64_t>> buckets(num_dpus);

    for (unsigned id=0;id< num_dpus; ++id) {

      if (shards_[id].state != CopyState::HOST_DIRTY)
        continue;

      if (serializer_) {
        buckets[shards_[id].blk.dpu] = serializer_(shards_[id].host);
      } 
      
      shards_[id].state = CopyState::CLEAN;
    }

    {
      std::lock_guard<std::mutex> lock(get_safe_console_mutex());
      // std::cout << "PIM: pushing " << buckets.size()
      //           << " buffers to DPUs, total size: "
      //           << std::accumulate(buckets.begin(), buckets.end(), 0ull,
      //                              [](size_t sum, const auto &p) {
      //                                return sum + p.first.second;
      //                              })
      //           << " bytes\n";
    }
    // Scatter the buffers to DPUs
    uint64_t off = shards_[0].blk.off; // Assuming all shards have the same offset
      mgr.scatter(buckets, DPU_MRAM_HEAP_POINTER_NAME, off);
  }

  void invalidate_host() const {
    for (auto &s : shards_)
      s.state = CopyState::PIM_FRESH;
  }

private:
  Serializer serializer_{nullptr};
  Deserializer deserializer_{nullptr};
  std::vector<Shard> shards_;
  size_t total_{0};
  mutable std::mutex build_mutex_;

  // Helper method to deallocate all blocks (only called in destructor and assignment operators)
  void deallocate_all() {
    // Only deallocate if we have shards and PIMManager is still valid
    if (!shards_.empty()) {
      try {
        auto &mgr = PIMManager::instance();
        for (auto &shard : shards_) {
          mgr.deallocate(shard.blk);
        }
      } catch (...) {
        // If PIMManager is no longer valid (e.g., during shutdown), 
        // we can't deallocate but we still need to clear our containers
        // The underlying DPU system will clean up when it shuts down
      }
    }
    shards_.clear();
    total_ = 0;
  }

  void build(size_t n, const T &fill) {
    std::lock_guard<std::mutex> lock(build_mutex_);
    auto &mgr = PIMManager::instance();
    if (mgr.num_dpus() == 0) PIMManager::init();

    total_ = n;
    unsigned D = mgr.num_dpus();
    const size_t chunk = (n + D - 1) / D;       // ceil(n/D) – every DPU gets this many
    size_t produced = 0;

    shards_.clear();
    for (unsigned d = 0; d < D; ++d) {
        size_t cnt = std::min(chunk, n - produced);   // real data for this DPU
        produced += cnt;

        Shard s;
        s.blk  = mgr.allocate(d,chunk * sizeof(uint64_t));     // reserve full chunk in MRAM
        s.host.assign(chunk, T{});                    // host buffer == chunk
        std::fill(s.host.begin(), s.host.begin()+cnt, fill); // initial data
        // any positions [cnt, chunk) are already zero-initialised -> padding
        shards_.push_back(std::move(s));
    }
}


// void build(size_t n, const T &fill) {
//     try {
//         auto &mgr = PIMManager::instance();
//         if (mgr.num_dpus() == 0) PIMManager::init();

//         total_ = n;
        
//         if (n == 0) {
//             // For empty vectors, just clear host data but keep allocated blocks
//             for (auto &shard : shards_) {
//                 shard.host.clear();
//                 shard.state = CopyState::HOST_DIRTY;
//             }
//             return;
//         }

//         unsigned D = mgr.num_dpus();
//         if (D == 0) {
//             throw std::runtime_error("No DPUs available");
//         }
        
//         const size_t chunk = (n + D - 1) / D;
//         size_t produced = 0;

//         // If we have no existing shards, create new ones
//         if (shards_.empty()) {
//             shards_.reserve(D);
//             for (unsigned d = 0; d < D && produced < n; ++d) {
//                 size_t cnt = std::min(chunk, n - produced);
//                 if (cnt == 0) break;
                
//                 produced += cnt;

//                 Shard s;
//                 s.blk = mgr.allocate(chunk * sizeof(T));
//                 s.host.assign(chunk, T{});
//                 if (cnt > 0) {
//                     std::fill(s.host.begin(), s.host.begin()+cnt, fill);
//                 }
//                 s.state = CopyState::HOST_DIRTY;
//                 shards_.push_back(std::move(s));
//             }
//         } else {
//             // Reuse existing shards, resize host data as needed
//             produced = 0;
//             for (size_t d = 0; d < shards_.size() && produced < n; ++d) {
//                 size_t cnt = std::min(chunk, n - produced);
//                 if (cnt == 0) {
//                     shards_[d].host.clear();
//                 } else {
//                     shards_[d].host.assign(chunk, T{});
//                     std::fill(shards_[d].host.begin(), shards_[d].host.begin()+cnt, fill);
//                     produced += cnt;
//                 }
//                 shards_[d].state = CopyState::HOST_DIRTY;
//             }
            
//             // If we need more shards, allocate them
//             while (produced < n && shards_.size() < D) {
//                 size_t cnt = std::min(chunk, n - produced);
//                 produced += cnt;

//                 Shard s;
//                 s.blk = mgr.allocate(chunk * sizeof(T));
//                 s.host.assign(chunk, T{});
//                 if (cnt > 0) {
//                     std::fill(s.host.begin(), s.host.begin()+cnt, fill);
//                 }
//                 s.state = CopyState::HOST_DIRTY;
//                 shards_.push_back(std::move(s));
//             }
//         }
//     } catch (...) {
//         // If build fails, ensure we're in a consistent state
//         total_ = 0;
//         throw;
//     }
// }

  std::pair<size_t, size_t> locate(size_t i) const {
    if (i >= total_)
      throw std::out_of_range("Vector index");
    size_t off = 0;
    for (size_t s = 0; s < shards_.size(); ++s) {
      size_t cnt = shards_[s].host.size();
      if (i < off + cnt)
        return {s, i - off};
      off += cnt;
    }
    throw std::runtime_error("locate failed");
  }

  void pull_all() const {
    auto &mgr = PIMManager::instance();

    size_t bytes = 0;
    bool any_fresh = false;

    for (const auto &sh : shards_) {
      if (sh.state == CopyState::PIM_FRESH) {
        bytes = sh.blk.bytes; // assume uniform size
        any_fresh = true;
        break;
      }
    }

    if (!any_fresh)
      return;
    {
      std::lock_guard<std::mutex> lock(get_safe_console_mutex());
      // std::cout << "PIM: pulling " << bytes << " bytes from "
      //           << shards_.size() << " shards\n";
    }
    std::vector<std::vector<uint64_t>> per_dpu;
    mgr.gather(per_dpu, bytes, shards_[0].blk.off); // same offset for all

    for (auto &sh : shards_) {
      if (sh.state == CopyState::PIM_FRESH) {
        // Use custom deserializer if available, otherwise use raw bytes
        if (deserializer_) {
          deserializer_(per_dpu[sh.blk.dpu], const_cast<std::vector<T>&>(sh.host));
        } else {
          // Cast away const safely since we're modifying the data in a logical const method
          void* dest_ptr = const_cast<void*>(static_cast<const void*>(sh.host.data()));
          std::memcpy(dest_ptr, per_dpu[sh.blk.dpu].data(), sh.blk.bytes);
        }
        sh.state = CopyState::CLEAN;
      }
    }
  }
};

/* run_kernel */
template <typename... InBufs, typename... OutBufs>
void run_kernel(const dpu_arguments_t &args, std::tuple<InBufs &...> in,
                std::tuple<OutBufs &...> out) {
                  
  /* push dirty inputs */
  std::apply([](auto &...bs) { (bs.commit(), ...); }, in);

  /* broadcast WRAM args, launch, mark outputs stale */
  auto &mgr = PIMManager::instance();
  mgr.push_args(&args, sizeof(args));
  mgr.exec();
  std::apply([](auto &...bs) { (bs.invalidate_host(), ...); }, out);
}
} // namespace pim