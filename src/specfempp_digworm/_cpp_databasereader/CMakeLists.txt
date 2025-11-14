
if(__TERRARIUM_LINKING__)

  message(STATUS "Compiling database reader.")

  python_add_library(_databasereader MODULE module.cpp WITH_SOABI)

  target_link_libraries(
    _databasereader
    PRIVATE
    pybind11::headers 
    Kokkos::kokkos
    ${TERRARIUM_ARCHIVES}/libquadrature.a
    ${TERRARIUM_ARCHIVES}/libio.a
  )
  
  # libedge.a
  # libenumerations.a
  # libjacobian.a
  # libkokkosalgorithms.a
  # libkokkoscontainers.a
  # libkokkoscore.a
  # libkokkossimd.a
  # libpoint.a
  # libquadrature.a
  # libread_seismogram.a
  # libreceiver_class.a
  # libshape_functions.a
  # libspecfem_mpi.a
  # libutilities.a
  # libyaml-cpp.a

  target_compile_definitions(_databasereader PRIVATE VERSION_INFO=${PROJECT_VERSION})
  install(TARGETS _databasereader DESTINATION specfempp_digworm)
else()

  message(STATUS "Skipping database reader.")
endif()