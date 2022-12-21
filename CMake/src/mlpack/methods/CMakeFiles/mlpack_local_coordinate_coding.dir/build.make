# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/adarsh/gsoc/mlpack

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/adarsh/gsoc/mlpack/CMake

# Include any dependencies generated for this target.
include src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/depend.make

# Include the progress variables for this target.
include src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/progress.make

# Include the compile flags for this target's objects.
include src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/flags.make

src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.o: src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/flags.make
src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.o: ../src/mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/adarsh/gsoc/mlpack/CMake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.o"
	cd /home/adarsh/gsoc/mlpack/CMake/src/mlpack/methods && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.o -c /home/adarsh/gsoc/mlpack/src/mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp

src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.i"
	cd /home/adarsh/gsoc/mlpack/CMake/src/mlpack/methods && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/adarsh/gsoc/mlpack/src/mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp > CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.i

src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.s"
	cd /home/adarsh/gsoc/mlpack/CMake/src/mlpack/methods && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/adarsh/gsoc/mlpack/src/mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp -o CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.s

# Object files for target mlpack_local_coordinate_coding
mlpack_local_coordinate_coding_OBJECTS = \
"CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.o"

# External object files for target mlpack_local_coordinate_coding
mlpack_local_coordinate_coding_EXTERNAL_OBJECTS =

bin/mlpack_local_coordinate_coding: src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/local_coordinate_coding/local_coordinate_coding_main.cpp.o
bin/mlpack_local_coordinate_coding: src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/build.make
bin/mlpack_local_coordinate_coding: /usr/lib/libarmadillo.so
bin/mlpack_local_coordinate_coding: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
bin/mlpack_local_coordinate_coding: /usr/lib/x86_64-linux-gnu/libpthread.so
bin/mlpack_local_coordinate_coding: src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/adarsh/gsoc/mlpack/CMake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/mlpack_local_coordinate_coding"
	cd /home/adarsh/gsoc/mlpack/CMake/src/mlpack/methods && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mlpack_local_coordinate_coding.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/build: bin/mlpack_local_coordinate_coding

.PHONY : src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/build

src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/clean:
	cd /home/adarsh/gsoc/mlpack/CMake/src/mlpack/methods && $(CMAKE_COMMAND) -P CMakeFiles/mlpack_local_coordinate_coding.dir/cmake_clean.cmake
.PHONY : src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/clean

src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/depend:
	cd /home/adarsh/gsoc/mlpack/CMake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/adarsh/gsoc/mlpack /home/adarsh/gsoc/mlpack/src/mlpack/methods /home/adarsh/gsoc/mlpack/CMake /home/adarsh/gsoc/mlpack/CMake/src/mlpack/methods /home/adarsh/gsoc/mlpack/CMake/src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/mlpack/methods/CMakeFiles/mlpack_local_coordinate_coding.dir/depend

