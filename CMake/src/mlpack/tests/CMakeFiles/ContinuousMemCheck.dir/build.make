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

# Utility rule file for ContinuousMemCheck.

# Include the progress variables for this target.
include src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/progress.make

src/mlpack/tests/CMakeFiles/ContinuousMemCheck:
	cd /home/adarsh/gsoc/mlpack/CMake/src/mlpack/tests && /usr/bin/ctest -D ContinuousMemCheck

ContinuousMemCheck: src/mlpack/tests/CMakeFiles/ContinuousMemCheck
ContinuousMemCheck: src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/build.make

.PHONY : ContinuousMemCheck

# Rule to build all files generated by this target.
src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/build: ContinuousMemCheck

.PHONY : src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/build

src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/clean:
	cd /home/adarsh/gsoc/mlpack/CMake/src/mlpack/tests && $(CMAKE_COMMAND) -P CMakeFiles/ContinuousMemCheck.dir/cmake_clean.cmake
.PHONY : src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/clean

src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/depend:
	cd /home/adarsh/gsoc/mlpack/CMake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/adarsh/gsoc/mlpack /home/adarsh/gsoc/mlpack/src/mlpack/tests /home/adarsh/gsoc/mlpack/CMake /home/adarsh/gsoc/mlpack/CMake/src/mlpack/tests /home/adarsh/gsoc/mlpack/CMake/src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/mlpack/tests/CMakeFiles/ContinuousMemCheck.dir/depend

