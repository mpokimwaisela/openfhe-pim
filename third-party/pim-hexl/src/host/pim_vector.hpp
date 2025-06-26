/**
 * @file pim_vector.hpp
 * @brief Main header for PIM (Processing-In-Memory) host-side operations
 * 
 * This file provides the primary interface for PIM operations including:
 * - PIM Vector container with automatic DPU synchronization
 * - Template-based kernel execution framework
 * - STL-compatible iterators and container operations
 * - Custom serialization support for complex data types
 * - Thread-safe operations with proper locking
 */

#pragma once

// Standard library includes
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#include <type_traits>  // Add this line for std::is_same_v


// Project-specific includes
#include "pim_launch_args.hpp"
#include "pim_vector_iterator.hpp"
#include "mram_allocator.hpp"
#include "pim_manager.hpp"
#include "profiler.hpp"

namespace pim {

/**
 * @typedef ShardedVector
 * @brief A vector distributed across DPU shards
 * @tparam T The element type stored in the vector
 */
template <typename T>
using ShardedVector = std::vector<std::vector<T>>;

/**
 * @enum CopyState
 * @brief Tracks the synchronization state of data between host and DPU
 * 
 * This enum is used to manage data coherency between host memory and DPU MRAM.
 * It ensures that data is properly synchronized when accessed from either side.
 */
enum class CopyState : uint8_t { 
    CLEAN,      ///< Data is synchronized between host and DPU
    HOST_DIRTY, ///< Host has newer data that needs to be sent to DPU
    PIM_FRESH   ///< DPU has newer data that needs to be pulled to host
};

/**
 * @class Vector
 * @brief Distributed vector container that spans across multiple DPUs
 * @tparam T Element type stored in the vector
 * 
 * The Vector class provides an STL-compatible container interface while
 * automatically managing data distribution across multiple DPUs. Key features:
 * - Automatic data sharding across available DPUs
 * - Lazy synchronization between host and DPU memory
 * - STL-compatible iterators and operations
 * - Custom serialization support for complex data types
 * - Thread-safe operations with proper locking
 */
template <typename T> 
class Vector {
private:
    // /**
    //  * @struct Shard
    //  * @brief Represents a portion of the vector stored on a single DPU
    //  */

public:
    // Type aliases for STL compatibility
    using Serializer   = std::function<ShardedVector<dpu_word_t>(const ShardedVector<T>&)>;
    using Deserializer = std::function<void(const ShardedVector<dpu_word_t>&, ShardedVector<T>&)>;
    using iterator = VectorIterator<T>;
    using const_iterator = VectorConstIterator<T>;

    // ═══════════════════════════════════════════════════════════════════════════════════
    // CONSTRUCTORS AND DESTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════════════

    Vector() : shard_{0, 0}, total_(0) {}
    
    ~Vector() {
        if (shutdown_mode.load()) {
            return;  // Skip cleanup during shutdown  
    }
        try {
            deallocate_if_owner();
        } catch (...) {
            // Ignore exceptions during destruction to prevent termination issues
        }
    }
    
    Vector(const Vector& other) : shard_{0, 0}, total_(0) { 
        *this = other; 
    }
    
    Vector(Vector&& other) noexcept : shard_{0, 0}, total_(0) { 
        *this = std::move(other); 
    }
    
    explicit Vector(size_t n) : shard_{0, 0}, total_(0) { 
        build(n, T{}); 
    }
    
    Vector(size_t n, const T &v) : shard_{0, 0}, total_(0) { 
        build(n, v); 
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // ASSIGNMENT OPERATORS
    // ═══════════════════════════════════════════════════════════════════════════════════

    Vector& operator=(const Vector& other) {
        if (this != &other) {
            // Clean up current allocation
            deallocate_if_owner();
            
            // Copy other's state
            serializer_ = other.serializer_;
            deserializer_ = other.deserializer_;
            
            if (other.total_ > 0) {
                // Ensure other's data is on host
                other.pull_all();
                
                // Allocate our own memory
                build(other.total_, T{});
                
                // Copy shards data directly instead of element-by-element
                for (size_t s = 0; s < shards_.size() && s < other.shards_.size(); ++s) {
                    for (size_t i = 0; i < shards_[s].size() && i < other.shards_[s].size(); ++i) {
                        shards_[s][i] = other.shards_[s][i];
                    }
                }
                state_ = CopyState::HOST_DIRTY;
            } else {
                reset_to_empty();
            }
        }
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            // Clean up current allocation
            deallocate_if_owner();
            
            // Move other's resources
            total_ = other.total_;
            shard_ = other.shard_;
            serializer_ = std::move(other.serializer_);
            deserializer_ = std::move(other.deserializer_);
            shards_ = std::move(other.shards_);
            state_ = other.state_;
            
            // Reset other to empty state (important: don't deallocate!)
            other.reset_to_empty();
        }
        return *this;
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SERIALIZATION CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Set custom serializer for complex data types
     * @param s Serializer function that converts vector<T> to vector<dpu_word_t>
     */
    void set_serializer(Serializer s) { serializer_ = s; }

    /**
     * @brief Set custom deserializer for complex data types
     * @param d Deserializer function that converts vector<dpu_word_t> to vector<T>
     */
    void set_deserializer(Deserializer d) { deserializer_ = d; }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // STL-COMPATIBLE INTERFACE
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Get the number of elements in the vector
     * @return Number of elements
     */
    size_t size() const { return total_; }

    /**
     * @brief Check if the vector is empty
     * @return true if vector has no elements, false otherwise
     */
    bool empty() const { return total_ == 0; }

    /**
     * @brief Remove all elements from the vector
     */
    void clear() { 
        deallocate_if_owner();
        reset_to_empty();
    }

    /**
     * @brief Resize the vector to contain specified number of elements
     * @param new_size New size of the vector
     */
    void resize(size_t new_size) { 
        if (new_size != total_) build(new_size, T{}); 
    }

    /**
     * @brief Resize the vector with a specific fill value
     * @param new_size New size of the vector
     * @param value Value to fill new elements with
     */
    void resize(size_t new_size, const T& value) { 
        if (new_size != total_) build(new_size, value); 
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // ITERATOR SUPPORT
    // ═══════════════════════════════════════════════════════════════════════════════════

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, total_); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, total_); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, total_); }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // ELEMENT ACCESS
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Access element with bounds checking
     * @param index Element index
     * @return Reference to the element
     * @throws std::out_of_range if index is out of bounds
     */
    T& at(size_t index) {
        if (index >= total_) throw std::out_of_range("Vector index out of range");
        return (*this)[index];
    }

    const T& at(size_t index) const {
        if (index >= total_) throw std::out_of_range("Vector index out of range");
        return (*this)[index];
    }

    /**
     * @brief Access element without bounds checking
     * @param i Element index
     * @return Reference to the element
     */
    T &operator[](size_t i) {
        pull_all();
        std::pair<size_t, size_t> loc = locate(i);
        size_t s = loc.first;
        size_t idx = loc.second;
        state_ = CopyState::HOST_DIRTY;
        return shards_[s][idx];
    } // need to check if this alway means assignment to the vector?

    const T &operator[](size_t i) const{
        pull_all();
        std::pair<size_t, size_t> loc = locate(i);
        size_t s = loc.first;
        size_t idx = loc.second;
        return shards_[s][idx];
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SERIALIZATION SUPPORT
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Save vector to archive (Cereal serialization support)
     * @tparam Archive Archive type
     * @param ar Archive to save to
     */
    template <class Archive>
    void save(Archive& ar) const {
        pull_all(); // Ensure all data is on host
        ar(total_);
        for (size_t i = 0; i < total_; ++i) {
            ar((*this)[i]);
        }
    }

    /**
     * @brief Load vector from archive (Cereal serialization support)
     * @tparam Archive Archive type
     * @param ar Archive to load from
     */
    template <class Archive>
    void load(Archive& ar) {
        size_t new_size;
        ar(new_size);
        if (new_size != total_) {
            resize(new_size);
        }
        for (size_t i = 0; i < new_size; ++i) {
            T value;
            ar(value);
            (*this)[i] = value;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // DPU SYNCHRONIZATION
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Commit host changes to DPU memory
     * 
     * Transfers any dirty data from host memory to DPU MRAM.
     * This is automatically called when needed, but can be manually invoked
     * to ensure data is synchronized before DPU kernel execution.
     */
    void commit() const {
        PROFILE_FUNCTION();
        if (state_ != CopyState::HOST_DIRTY) return;
        auto &mgr = PIMManager::instance();
        // std::cout << "Committing " << shards_.size() << " shards to DPU memory\n";

        if constexpr (std::is_same_v<T, dpu_word_t>) {
            // T is dpu_word_t, can use direct scatter
            mgr.scatter(shards_, shard_.off);
        } else {
            // T is not dpu_word_t, must use serializer
            if (serializer_ != nullptr) {
                auto serialized = serializer_(shards_);
                mgr.scatter(serialized, shard_.off);
            } else {
                throw std::runtime_error("Serializer required for non-dpu_word_t types");
            }
        }
        state_ = CopyState::CLEAN;
    }
    /**
     * @brief Mark host data as stale (DPU has newer data)
     * 
     * This is called after DPU kernel execution to indicate that
     * the DPU may have modified the data and host copies are stale.
     */

    void invalidate_host() const { state_ = CopyState::PIM_FRESH; }


    /**
     * @brief Get access to internal shard (for debugging)
     * @return Const reference to shards vector
     */
    const PIMManager::Block &shard() const { return shard_; } 

private:
    // ═══════════════════════════════════════════════════════════════════════════════════
    // PRIVATE MEMBER VARIABLES
    // ═══════════════════════════════════════════════════════════════════════════════════

    Serializer serializer_{nullptr};       ///< Custom serializer function
    Deserializer deserializer_{nullptr};   ///< Custom deserializer function
    PIMManager::Block shard_;             ///< Data shards distributed across DPUs
    mutable ShardedVector<T> shards_;             ///< Data shards distributed across DPUs
    size_t total_{0};                       ///< Total number of elements
    mutable CopyState state_{CopyState::HOST_DIRTY}; ///< Synchronization state

    // ═══════════════════════════════════════════════════════════════════════════════════
    // PRIVATE HELPER METHODS
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Safely deallocate DPU memory if this object owns it
     */
    void deallocate_if_owner() {
        if (total_ > 0 && shard_.bytes > 0) {
            try {
                auto &mgr = PIMManager::instance();
                mgr.deallocate(shard_);
            } catch (...) {
                // Ignore errors during shutdown or if manager is invalid
            }
        }
    }

    /**
     * @brief Reset object to empty state without deallocating
     */
    void reset_to_empty() {
        shard_ = {0, 0};
        total_ = 0;
        shards_.clear();
        state_ = CopyState::HOST_DIRTY;
        serializer_ = nullptr;
        deserializer_ = nullptr;
    }

    /**
     * @brief Deallocate all DPU memory blocks
     */
    void deallocate_all() {
        deallocate_if_owner();
        reset_to_empty();
    }

    /**
     * @brief build vector with specified size and fill value
     * @param n Number of elements
     * @param fill Value to fill elements with
     */
    void build(size_t n, const T &fill) {
      PROFILE_FUNCTION();
      auto &mgr = PIMManager::instance();
      if (mgr.num_dpus() == 0) PIMManager::init();

      // Clean up existing allocation first
      deallocate_if_owner();

      if (n == 0) {
        reset_to_empty();
        return;
      }

      total_ = n;
      unsigned D = mgr.num_dpus();
      size_t chunk_ = (n + D - 1) / D;
      
      if (chunk_ % 8) chunk_ += 8 - (chunk_ % 8);

      size_t bytes_per_chunk = chunk_ * sizeof(T);
      
      size_t aligned_bytes = ((bytes_per_chunk + 7) / 8) * 8;
      
      shard_ = mgr.allocate_uniform(aligned_bytes);
      // std::cout << "Allocated " << chunk_ << " elements (" << aligned_bytes << " aligned bytes) on each DPU shard from offset "<< shard_.off << " [ptr=" << (void*)this << "]" << std::endl;
      
      shards_.assign(D, std::vector<T>(chunk_, fill));
      state_ = CopyState::HOST_DIRTY;
    }


    /**
     * @brief Locate which shard and index within shard for global index
     * @param i Global element index
     * @return Pair of (shard_index, local_index)
     */
    std::pair<size_t, size_t> locate(size_t i) const {
        if (i >= total_)
            throw std::out_of_range("Vector index out of range");
        
        if (shards_.empty())
            throw std::runtime_error("Vector has no shards allocated");
            
        // Calculate which shard and offset based on uniform distribution
        size_t elements_per_shard = shards_[0].size();
        size_t shard_index = i / elements_per_shard;
        size_t local_index = i % elements_per_shard;
        
        if (shard_index >= shards_.size())
            throw std::runtime_error("Shard index out of range");
            
        // Handle the case where the last shard might have fewer elements
        if (shard_index == shards_.size() - 1 && local_index >= shards_[shard_index].size()) {
            throw std::out_of_range("Index beyond actual shard size");
        }
        
        return {shard_index, local_index};
    }

    /**
     * @brief Pull fresh data from DPU to host memory
     */
    void pull_all() const {
        PROFILE_FUNCTION();
        if (state_ != CopyState::PIM_FRESH) return;
        auto &mgr = PIMManager::instance();

        if constexpr (std::is_same_v<T, dpu_word_t>) {
             mgr.gather(shards_,shard_.bytes, shard_.off);
        } else {
            if (deserializer_ != nullptr) {
                ShardedVector<dpu_word_t> gathered(mgr.num_dpus(),std::vector<dpu_word_t>(shard_.bytes / sizeof(dpu_word_t)));
                mgr.gather(gathered, shard_.bytes, shard_.off);
                deserializer_(gathered, shards_);
            } else {
                throw std::runtime_error("Deserializer required for non-dpu_word_t types");
            }
        }
        state_ = CopyState::CLEAN;
    }
};
// ═══════════════════════════════════════════════════════════════════════════════════════
// KERNEL EXECUTION FRAMEWORK
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * @brief Execute a DPU kernel with input and output buffers
 * @tparam InBufs Input buffer types (PIM Vectors)
 * @tparam OutBufs Output buffer types (PIM Vectors)
 * @param args Kernel arguments structure
 * @param in Tuple of input buffers
 * @param out Tuple of output buffers
 * 
 * This function orchestrates the complete kernel execution pipeline:
 * 1. Commits all dirty input data to DPUs
 * 2. Broadcasts kernel arguments to all DPUs
 * 3. Executes the kernel on all DPUs
 * 4. Marks output buffers as having fresh data on DPUs
 */
template <typename... InBufs, typename... OutBufs>
void run_kernel(const dpu_arguments_t &args, std::tuple<InBufs &...> in,
                std::tuple<OutBufs &...> out) {
    // PROFILE_FUNCTION();
    std::apply([](auto &...bs) { (bs.commit(), ...); }, in);

    auto &mgr = PIMManager::instance();
    mgr.push_args(&args, sizeof(args));
    mgr.exec();
    std::apply([](auto &...bs) { (bs.invalidate_host(), ...); }, out);
}

} // namespace pim
