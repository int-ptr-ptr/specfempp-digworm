# ===================================================
#  SPECFEM++ 

# function(specfem_compile)

#   set(SPECFEM_ENABLE_VTK OFF)
#   set(Kokkos_ENABLE_SERIAL ON)
#   set(CMAKE_BUILD_TYPE Release)
#   SET(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON)
#   SET(SPECFEM_ENABLE_SIMD ON)
#   SET(Kokkos_ENABLE_ATOMICS_BYPASS OFF)

#   SET(CMAKE_C_COMPILER /usr/bin/gcc-13)
#   SET(CMAKE_CXX_COMPILER /usr/bin/g++-13)
#   SET(CMAKE_Fortran_COMPILER /usr/bin/gfortran-13)

#   add_subdirectory(SPECFEMPP)

# endfunction()

# specfem_compile()
# ===================================================


# can we catch this from build?
if (SPECFEM_ENABLE_DOUBLE_PRECISION)
  message(STATUS "Enabling double precision")
  set(TYPE_REAL "double")
else(SPECFEM_ENABLE_DOUBLE_PRECISION)
  message(STATUS "Enabling single precision")
  set(TYPE_REAL "float")
endif(SPECFEM_ENABLE_DOUBLE_PRECISION)

set(CHUNK_SIZE 32)
set(NUM_CHUNKS 1)
set(NUM_THREADS 160)
set(NUM_VECTOR_LANES 1)

# Configure the setup headers
configure_file(${TERRARIUM_REPO_DIR}/setup/specfem_setup.hpp.in
               ${TERRARIUM_REPO_BINARY_DIR}/include/specfem_setup.hpp)
configure_file(${TERRARIUM_REPO_DIR}/setup/constants.hpp.in
               ${TERRARIUM_REPO_BINARY_DIR}/include/constants.hpp)


# This is a hacky change. We may want to replace this
configure_file(${TERRARIUM_REPO_DIR}/setup/specfem_setup.hpp.in
               ${TERRARIUM_REPO_DIR}/include/specfem_setup.hpp)
configure_file(${TERRARIUM_REPO_DIR}/setup/constants.hpp.in
               ${TERRARIUM_REPO_DIR}/include/constants.hpp)


# Add the configure files to the clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${TERRARIUM_REPO_BINARY_DIR}/include/specfem_setup.hpp
    ${TERRARIUM_REPO_BINARY_DIR}/include/constants.hpp
)