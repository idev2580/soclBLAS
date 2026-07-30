#!/bin/bash
echo "###### 1. Shared Memory ######"
./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 32 128 8 8 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 16 128 8 8 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 256 8 64 8 8 20

echo "###### 2. Tile reusage ######"
./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 8 8 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 128 8 8 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 32 64 8 8 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 32 128 8 8 20

echo "###### 3. Register occupancy ######"
./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 4 4 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 8 4 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 4 8 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 8 8 20

echo "###### 4. Axis ######"
./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 16 256 8 8 20
./build/soclblas_performance_tests --cont 8 4096 1024 1024 256 16 64 8 8 20