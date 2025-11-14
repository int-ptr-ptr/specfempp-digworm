cmake_minimum_required(VERSION 4.0...4.1)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

find_package(pybind11 CONFIG REQUIRED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
endif()

# include_directories(${CMAKE_SOURCE_DIR}/include)

if(EXISTS "${CMAKE_SOURCE_DIR}/terrarium/.LINKABLE")
  set(__TERRARIUM_LINKING__ TRUE)

  set(TERRARIUM_REPO_DIR ${CMAKE_SOURCE_DIR}/terrarium/SPECFEMPP)
  set(TERRARIUM_REPO_BINARY_DIR ${CMAKE_BINARY_DIR}/terrarium/SPECFEMPP)
  set(TERRARIUM_BUILD_DIR ${TERRARIUM_REPO_DIR}/build)
  set(TERRARIUM_ARCHIVES ${TERRARIUM_BUILD_DIR}/archive)


  # options needed for compiling
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  SET(CMAKE_C_COMPILER /usr/bin/gcc-13)
  SET(CMAKE_CXX_COMPILER /usr/bin/g++-13)
  SET(CMAKE_Fortran_COMPILER /usr/bin/gfortran-13)

  # For the installation of dependencies
  include(FetchContent)

  # For external modules that need to be built suppress build output
  set(FETCHCONTENT_QUIET TRUE)

  include("${TERRARIUM_REPO_DIR}/cmake/kokkos.cmake")
  include("${TERRARIUM_REPO_DIR}/cmake/yaml.cmake")
  include("${TERRARIUM_REPO_DIR}/cmake/boost.cmake")
  include("${TERRARIUM_REPO_DIR}/cmake/hdf5.cmake")
  include("${TERRARIUM_REPO_DIR}/cmake/npz.cmake")
  include("${TERRARIUM_REPO_DIR}/cmake/adios2.cmake")
  include("${TERRARIUM_REPO_DIR}/cmake/vtk.cmake")

  include_directories(${TERRARIUM_REPO_DIR}/core)
  include_directories(${TERRARIUM_REPO_DIR}/include)


  message(STATUS "Found SPECFEM++ repository folder. Adding to CMake")
  add_subdirectory(terrarium)

else()
  set(__TERRARIUM_LINKING__ FALSE)
  message(STATUS "SPECFEM++ repository folder not found. Skipping worm features.")
endif()


add_subdirectory(src/specfempp_digworm/_cpp_databasereader)