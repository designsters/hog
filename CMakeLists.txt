cmake_minimum_required (VERSION 2.6)

project (cv_task)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
	set(CMAKE_EXE_LINKER_FLAGS "-s")
endif()

SET(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

SET(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

SET(DARKNET_SRC_DIR ${PROJECT_SOURCE_DIR}/lib/darknet/src)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}/src)

add_executable(cv_task ${SRC_DIR}/main.cpp 
			${SRC_DIR}/detector.cpp 
			${SRC_DIR}/detector.h 
			${DARKNET_SRC_DIR}/network.c 
			${DARKNET_SRC_DIR}/matrix.c
			${DARKNET_SRC_DIR}/image.c
			${DARKNET_SRC_DIR}/data.c
			${DARKNET_SRC_DIR}/parser.c
			${DARKNET_SRC_DIR}/utils.c
			${DARKNET_SRC_DIR}/list.c
			${DARKNET_SRC_DIR}/option_list.c
			${DARKNET_SRC_DIR}/blas.c
			${DARKNET_SRC_DIR}/layer.c
			${DARKNET_SRC_DIR}/convolutional_layer.c
			${DARKNET_SRC_DIR}/deconvolutional_layer.c
			${DARKNET_SRC_DIR}/normalization_layer.c
			${DARKNET_SRC_DIR}/batchnorm_layer.c
			${DARKNET_SRC_DIR}/activation_layer.c
			${DARKNET_SRC_DIR}/local_layer.c
			${DARKNET_SRC_DIR}/detection_layer.c
			${DARKNET_SRC_DIR}/connected_layer.c
			${DARKNET_SRC_DIR}/rnn_layer.c
			${DARKNET_SRC_DIR}/gru_layer.c
			${DARKNET_SRC_DIR}/crnn_layer.c
			${DARKNET_SRC_DIR}/crop_layer.c
			${DARKNET_SRC_DIR}/cost_layer.c
			${DARKNET_SRC_DIR}/softmax_layer.c
			${DARKNET_SRC_DIR}/maxpool_layer.c
			${DARKNET_SRC_DIR}/avgpool_layer.c
			${DARKNET_SRC_DIR}/dropout_layer.c
			${DARKNET_SRC_DIR}/route_layer.c
			${DARKNET_SRC_DIR}/shortcut_layer.c
			${DARKNET_SRC_DIR}/gemm.c
			${DARKNET_SRC_DIR}/box.c
			${DARKNET_SRC_DIR}/im2col.c
			${DARKNET_SRC_DIR}/col2im.c
			${DARKNET_SRC_DIR}/activations.c)

TARGET_LINK_LIBRARIES(cv_task pthread)

INSTALL(TARGETS cv_task DESTINATION ${PROJECT_SOURCE_DIR})


