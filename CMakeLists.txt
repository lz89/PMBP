cmake_minimum_required(VERSION 2.8) 
PROJECT (PatchMatch)

#set(VTK_DIR "C:/Program Files (x86)/VTK/")

find_package(OpenCV REQUIRED)
#find_package(VTK REQUIRED)

set( NAME_SRC
    main.cpp
	PatchMatch.cpp
	CostFunction.cpp
	Patch.cpp
	Stereo.cpp
)

set( NAME_HEADERS
	PatchMatch.h
	CostFunction.h
	Patch.h
	Stereo.h
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
#include(${VTK_USE_FILE})
add_executable( PatchMatch ${NAME_SRC} ${NAME_HEADERS} )
target_link_libraries( PatchMatch ${OpenCV_LIBS} #${VTK_LIBRARIES}
)