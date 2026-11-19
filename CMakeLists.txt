cmake_minimum_required(VERSION 3.8)
project(topsbot_ppocr)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(topsbot_ai_infer REQUIRED)
find_package(OpenCV REQUIRED)

if(CMAKE_SYSROOT)
  set(TOPSBOT_PPOCR_SYSTEM_ROOT "${CMAKE_SYSROOT}/usr/local")
else()
  set(TOPSBOT_PPOCR_SYSTEM_ROOT "/usr/local")
endif()
set(TOPSBOT_PPOCR_SYSTEM_INCLUDE "${TOPSBOT_PPOCR_SYSTEM_ROOT}/include")
set(TOPSBOT_PPOCR_SYSTEM_LIB "${TOPSBOT_PPOCR_SYSTEM_ROOT}/lib")

add_library(topsbot_ppocr_utils SHARED
  src/postprocess.cpp
  src/ppocr_utils.cpp
  src/clipper.cpp
)
target_include_directories(topsbot_ppocr_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${TOPSBOT_PPOCR_SYSTEM_INCLUDE}
)
target_link_libraries(topsbot_ppocr_utils ${OpenCV_LIBS})
ament_target_dependencies(topsbot_ppocr_utils topsbot_ai_infer)

add_library(topsbot_ppocr_backend SHARED
  src/ppocr_backend.cpp
)
target_include_directories(topsbot_ppocr_backend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${TOPSBOT_PPOCR_SYSTEM_INCLUDE}
)
target_link_directories(topsbot_ppocr_backend PUBLIC
  ${TOPSBOT_PPOCR_SYSTEM_LIB}
)
target_link_libraries(topsbot_ppocr_backend
  topsbot_ppocr_utils
  ta_runtime
  ${OpenCV_LIBS}
)
ament_target_dependencies(topsbot_ppocr_backend
  pluginlib
  topsbot_ai_infer
)

add_executable(ppocr_file_cli
  src/ppocr_file_cli.cpp
)
target_include_directories(ppocr_file_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppocr_file_cli ${OpenCV_LIBS})
ament_target_dependencies(ppocr_file_cli
  ament_index_cpp
  topsbot_ai_infer
)

install(TARGETS
  topsbot_ppocr_utils
  topsbot_ppocr_backend
  ppocr_file_cli
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY imgs DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/ensure_models.sh
  scripts/model_manager.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

pluginlib_export_plugin_description_file(topsbot_ai_infer ppocr_plugins.xml)

# Testing disabled until test files are created
# if(BUILD_TESTING)
#   find_package(ament_cmake_gtest REQUIRED)
#   ament_add_gtest(test_ppocr_utils test/test_ppocr_utils.cpp)
#   if(TARGET test_ppocr_utils)
#     target_include_directories(test_ppocr_utils PRIVATE
#       ${TOPSBOT_PPOCR_SYSTEM_INCLUDE}
#     )
#     target_link_libraries(test_ppocr_utils topsbot_ppocr_utils)
#     ament_target_dependencies(test_ppocr_utils topsbot_ai_infer)
#   endif()
# endif()

ament_export_include_directories(include)
ament_export_libraries(
  topsbot_ppocr_utils
  topsbot_ppocr_backend)
ament_export_dependencies(pluginlib topsbot_ai_infer)
ament_package()
