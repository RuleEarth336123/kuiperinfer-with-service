cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

find_package(glog REQUIRED)
if (glog_FOUND)
    message(STATUS "Found glog: ${glog_VERSION}")
else()
    message(FATAL_ERROR "glog not found!")
endif()

set(WebService_INCLUDE_DIR "/home/chunyu123/github/KuiperInfer/service/TinyWebServer")
set(KuiperInfer_INCLUDE_DIR "/home/chunyu123/github/kuiperinfer-with-service/include")
set(JSON11_INCLUDE_DIRS "/usr/include/x86_64-linux-gnu")
set(JSON11_LIBRARIES "/usr/lib/x86_64-linux-gnu/libjson11.a")


# 添加包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${glog_INCLUDE_DIR}
    ${KuiperInfer_INCLUDE_DIR}
    ${JSON11_INCLUDE_DIRS}
    "/home/chunyu123/github/KuiperInfer/service"

)

# 添加链接目录
link_directories(${OpenCV_LIBRARY_DIRS} ${PROJECT_SOURCE_DIR}/lib "/usr/lib/x86_64-linux-gnu")

# 设置编译器优化标志
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif ()

# 查找 kuiper 库
find_library(KUIPER_LIB NAMES libkuiper.so PATHS "/home/chunyu123/github/KuiperInfer/lib" NO_DEFAULT_PATH)
if (NOT KUIPER_LIB)
    message(FATAL_ERROR "libKuiper library not found!")
endif()

# 添加可执行文件
add_executable(main main.cpp handle.cpp image_util.hpp image_util.cpp)

# 链接库文件
target_link_libraries(main 
    ${OpenCV_LIBS} 
    glog::glog 
    ${KUIPER_LIB} # 使用找到的 kuiper 库
    ${JSON11_LIBRARIES}
)

# 如果需要，可以设置目标的链接目录
target_link_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/lib)