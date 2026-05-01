#pragma once
#include <stdio.h>

#define LOG_INFO(fmt, ...) fprintf(stderr, "[info]  " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) fprintf(stderr, "[warn]  " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...)  fprintf(stderr, "[error] " fmt "\n", ##__VA_ARGS__)
#define LOG_SAVE(fmt, ...) fprintf(stderr, "[save]  " fmt "\n", ##__VA_ARGS__)
#define LOG_PROG(fmt, ...) fprintf(stderr, "\r" fmt,            ##__VA_ARGS__)
