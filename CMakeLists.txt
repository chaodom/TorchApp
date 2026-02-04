cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIB_NAME TorchApp)
project(${LIB_NAME})
set(CMAKE_DEBUG_POSTFIX d)
add_definitions(-DTORCHAPP_DLLEXPORT)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME} ../../torchvision/torchvision/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME} SRC_LIST)
add_library(${LIB_NAME} SHARED ${SRC_LIST})

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
target_link_libraries(${LIB_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

# torchvision
find_library(TORCHVISION_RELEASE_LIBRARY
    NAMES torchvision
    PATHS ../../torchvision/torchvision/lib
    NO_DEFAULT_PATH
)
find_library(TORCHVISION_DEBUG_LIBRARY
    NAMES torchvisiond
    PATHS ../../torchvision/torchvision/lib
    NO_DEFAULT_PATH
)
target_link_libraries(${LIB_NAME} $<$<CONFIG:Release>:${TORCHVISION_RELEASE_LIBRARY}>)
target_link_libraries(${LIB_NAME} $<$<CONFIG:Debug>:${TORCHVISION_DEBUG_LIBRARY}>)