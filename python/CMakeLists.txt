cmake_minimum_required(VERSION 3.12)

project(libint2-python)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall)

if (TARGET libint2_obj)
  set(libint2_python_target libint2_obj)
else()
  find_package(Libint2 REQUIRED)
  set(libint2_python_target Libint2::cxx)
endif()

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY      https://github.com/ValeevGroup/pybind11.git
  GIT_TAG             80d452484c5409444b0ec19383faa84bb7a4d351 # v2.4.3
  )
FetchContent_MakeAvailable(pybind11)

#find_package(Eigen3 3.3 REQUIRED)

pybind11_add_module(
  libint2-python MODULE
  EXCLUDE_FROM_ALL
  src/libint2/_libint2.cc
  src/libint2/engine.cc
  )

set_target_properties(
  libint2-python
  PROPERTIES
  #PREFIX ""
  OUTPUT_NAME _libint2
  )

target_link_libraries(libint2-python PUBLIC ${libint2_python_target})

# if (Eigen3::Eigen)
#   target_link_libraries(libint2-python INTERFACE Eigen3::Eigen)
# else()
#   include_directories(${EIGEN3_INCLUDE_DIR})
# endif()

#target_link_libraries(python-tiledarray INTERFACE Boost::boost)

configure_file(setup.py.in ${PROJECT_BINARY_DIR}/setup.py)

enable_testing()

# add the executable
add_test(
  NAME libint2-python
  COMMAND ${PYTHON_EXECUTABLE} -m setup test
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )

add_custom_target(
  bdist_wheel
  DEPENDS libint2-python
  COMMAND ${PYTHON_EXECUTABLE} -m setup bdist_wheel
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )
