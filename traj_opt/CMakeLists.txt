cmake_minimum_required(VERSION 3.12)
project(trajectory_optimization)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(ifopt REQUIRED)

# 添加可执行文件
add_executable(trajectory_optimization src/main.cpp)

# 链接库
target_link_libraries(trajectory_optimization
    Eigen3::Eigen
    pinocchio::pinocchio
    ifopt::ifopt_core
    ifopt::ifopt_ipopt
)


# Formulate (ifopt:ifopt_core) and solve (ifopt::ifopt_ipopt) the problem
add_executable(example src/simple_example.cpp)
# Pull in include directories, libraries, ... 
target_link_libraries(example PUBLIC ifopt::ifopt_ipopt) 

