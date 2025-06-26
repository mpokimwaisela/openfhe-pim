#pragma once

#include <iterator>

namespace pim {

// Forward declaration
template <typename T> class Vector;

// Iterator class for Vector
template <typename T>
class VectorIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    
    VectorIterator(Vector<T>* buffer, size_t index) : buffer_(buffer), index_(index) {}
    
    reference operator*() { return (*buffer_)[index_]; }
    pointer operator->() { return &((*buffer_)[index_]); }
    
    VectorIterator& operator++() { ++index_; return *this; }
    VectorIterator operator++(int) { VectorIterator tmp = *this; ++index_; return tmp; }
    VectorIterator& operator--() { --index_; return *this; }
    VectorIterator operator--(int) { VectorIterator tmp = *this; --index_; return tmp; }
    
    VectorIterator& operator+=(difference_type n) { index_ += n; return *this; }
    VectorIterator& operator-=(difference_type n) { index_ -= n; return *this; }
    VectorIterator operator+(difference_type n) const { return VectorIterator(buffer_, index_ + n); }
    VectorIterator operator-(difference_type n) const { return VectorIterator(buffer_, index_ - n); }
    
    difference_type operator-(const VectorIterator& other) const { return index_ - other.index_; }
    
    bool operator==(const VectorIterator& other) const { return index_ == other.index_; }
    bool operator!=(const VectorIterator& other) const { return index_ != other.index_; }
    bool operator<(const VectorIterator& other) const { return index_ < other.index_; }
    bool operator<=(const VectorIterator& other) const { return index_ <= other.index_; }
    bool operator>(const VectorIterator& other) const { return index_ > other.index_; }
    bool operator>=(const VectorIterator& other) const { return index_ >= other.index_; }
    
private:
    Vector<T>* buffer_;
    size_t index_;
};

// Const iterator class for Vector
template <typename T>
class VectorConstIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;
    
    VectorConstIterator(const Vector<T>* buffer, size_t index) : buffer_(buffer), index_(index) {}
    
    reference operator*() const { return (*buffer_)[index_]; }
    pointer operator->() const { return &((*buffer_)[index_]); }
    
    VectorConstIterator& operator++() { ++index_; return *this; }
    VectorConstIterator operator++(int) { VectorConstIterator tmp = *this; ++index_; return tmp; }
    VectorConstIterator& operator--() { --index_; return *this; }
    VectorConstIterator operator--(int) { VectorConstIterator tmp = *this; --index_; return tmp; }
    
    VectorConstIterator& operator+=(difference_type n) { index_ += n; return *this; }
    VectorConstIterator& operator-=(difference_type n) { index_ -= n; return *this; }
    VectorConstIterator operator+(difference_type n) const { return VectorConstIterator(buffer_, index_ + n); }
    VectorConstIterator operator-(difference_type n) const { return VectorConstIterator(buffer_, index_ - n); }
    
    difference_type operator-(const VectorConstIterator& other) const { return index_ - other.index_; }
    
    bool operator==(const VectorConstIterator& other) const { return index_ == other.index_; }
    bool operator!=(const VectorConstIterator& other) const { return index_ != other.index_; }
    bool operator<(const VectorConstIterator& other) const { return index_ < other.index_; }
    bool operator<=(const VectorConstIterator& other) const { return index_ <= other.index_; }
    bool operator>(const VectorConstIterator& other) const { return index_ > other.index_; }
    bool operator>=(const VectorConstIterator& other) const { return index_ >= other.index_; }
    
private:
    const Vector<T>* buffer_;
    size_t index_;
};

} // namespace pim
