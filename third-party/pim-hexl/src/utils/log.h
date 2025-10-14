#ifndef HEXL_UTILS_LOG_H
#define HEXL_UTILS_LOG_H
#include <stdio.h>

// Use PIM_ prefix to avoid conflicts with Google Test
#define PIM_COLOR_YELLOW "\033[1;33m"  // Bold yellow
#define PIM_COLOR_RED    "\033[1;31m"  // Bold red
#define PIM_COLOR_GREEN  "\033[1;32m"  // Bold green
#define PIM_COLOR_BLUE   "\033[1;34m"  // Bold blue
#define PIM_COLOR_RESET  "\033[0m"

#define LOG(color, fmt, ...) \
    printf(color fmt PIM_COLOR_RESET "\n", ##__VA_ARGS__)

#define LOG_INFO(fmt, ...)    LOG(PIM_COLOR_BLUE, "[ PIM INFO ] " fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)    LOG(PIM_COLOR_YELLOW, "[ PIM WARN ] " fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...)   LOG(PIM_COLOR_RED, "[ PIM ERROR ] " fmt, ##__VA_ARGS__)
#define LOG_SUCCESS(fmt, ...) LOG(PIM_COLOR_GREEN, "[ PIM SUCCESS ] " fmt, ##__VA_ARGS__)

#endif // HEXL_UTILS_LOG_H