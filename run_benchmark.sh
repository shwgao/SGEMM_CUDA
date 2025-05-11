#!/bin/bash

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <device_id> <kernel_numbers...>"
    echo "Example: $0 0 1 2 3 4 5"
    exit 1
fi

# 编译程序
echo "Compiling CUDA kernels..."
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake ..
cmake --build .
cd ..

# 检查编译是否成功
if [ ! -f "build/sgemm" ]; then
    echo "Compilation failed!"
    exit 1
fi

# 获取设备ID
DEVICE_ID=$1
shift

# 创建结果目录
RESULTS_DIR="benchmark_results"
mkdir -p $RESULTS_DIR

# 获取当前时间作为结果文件名的一部分
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_results.csv"

# 创建CSV文件头
echo "Kernel,Size,GFLOPs" > $RESULTS_FILE

# 测试的矩阵大小
SIZES=(128 256 512 1024 2048 4096)

# 运行每个kernel的测试
for kernel in "$@"; do
    echo "Testing kernel $kernel on device $DEVICE_ID"
    
    # 运行测试并获取输出
    output=$(DEVICE=$DEVICE_ID ./build/sgemm $kernel 2>&1)
    
    # 运行测试并提取GFLOPs
    for size in "${SIZES[@]}"; do
        echo "Testing size: $size"
        # 提取GFLOPs值
        performance_line=$(echo "$output" | grep "size: ($size)" | grep "performance:")
        if [ ! -z "$performance_line" ]; then
            # 提取GFLOPs值，匹配performance:和GFLOPS之间的数字
            gflops=$(echo "$performance_line" | grep -o 'performance:.*GFLOPS' | sed 's/performance: //' | sed 's/ GFLOPS//' | tr -d '()' | tr -d ' ')
            if [ ! -z "$gflops" ]; then
                echo "$kernel,$size,$gflops" >> $RESULTS_FILE
                echo "Recorded GFLOPs: $gflops"
            else
                echo "Warning: Could not extract GFLOPs value for size $size"
                echo "$performance_line"
            fi
        else
            echo "Warning: No performance data found for size $size"
            # 打印完整的输出以便调试
            echo "Full output for kernel $kernel:"
            echo "$output"
        fi
    done
done

# 检查是否有数据生成
if [ ! -s "$RESULTS_FILE" ]; then
    echo "Error: No benchmark data was collected!"
    exit 1
fi

# 使用conda环境运行Python脚本生成图表
echo "Generating performance plot..."
conda run -n learning python3 plot_results.py $RESULTS_FILE

echo "Benchmark completed. Results saved in $RESULTS_FILE"
echo "Performance plot saved in benchmark_results/performance_comparison.png" 