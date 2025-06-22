// Test file to verify direct inclusion of pim.hpp
#include <iostream>

// This should work directly without any integration headers
#include "pim.hpp"

int main() {
    std::cout << "Successfully included pim.hpp directly!" << std::endl;
    return 0;
}
