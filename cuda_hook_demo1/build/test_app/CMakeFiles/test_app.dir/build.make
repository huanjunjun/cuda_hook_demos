# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /root/cmake/cmake-3.17.3-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /root/cmake/cmake-3.17.3-Linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/cuda_hook_demos/cuda_hook_demo1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/cuda_hook_demos/cuda_hook_demo1/build

# Include any dependencies generated for this target.
include test_app/CMakeFiles/test_app.dir/depend.make

# Include the progress variables for this target.
include test_app/CMakeFiles/test_app.dir/progress.make

# Include the compile flags for this target's objects.
include test_app/CMakeFiles/test_app.dir/flags.make

test_app/CMakeFiles/test_app.dir/test_app.cu.o: test_app/CMakeFiles/test_app.dir/flags.make
test_app/CMakeFiles/test_app.dir/test_app.cu.o: ../test_app/test_app.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/cuda_hook_demos/cuda_hook_demo1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test_app/CMakeFiles/test_app.dir/test_app.cu.o"
	cd /root/cuda_hook_demos/cuda_hook_demo1/build/test_app && /usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/cuda_hook_demos/cuda_hook_demo1/test_app/test_app.cu -o CMakeFiles/test_app.dir/test_app.cu.o

test_app/CMakeFiles/test_app.dir/test_app.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_app.dir/test_app.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test_app/CMakeFiles/test_app.dir/test_app.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_app.dir/test_app.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_app
test_app_OBJECTS = \
"CMakeFiles/test_app.dir/test_app.cu.o"

# External object files for target test_app
test_app_EXTERNAL_OBJECTS =

test_app/test_app: test_app/CMakeFiles/test_app.dir/test_app.cu.o
test_app/test_app: test_app/CMakeFiles/test_app.dir/build.make
test_app/test_app: test_app/CMakeFiles/test_app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/cuda_hook_demos/cuda_hook_demo1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable test_app"
	cd /root/cuda_hook_demos/cuda_hook_demo1/build/test_app && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_app/CMakeFiles/test_app.dir/build: test_app/test_app

.PHONY : test_app/CMakeFiles/test_app.dir/build

test_app/CMakeFiles/test_app.dir/clean:
	cd /root/cuda_hook_demos/cuda_hook_demo1/build/test_app && $(CMAKE_COMMAND) -P CMakeFiles/test_app.dir/cmake_clean.cmake
.PHONY : test_app/CMakeFiles/test_app.dir/clean

test_app/CMakeFiles/test_app.dir/depend:
	cd /root/cuda_hook_demos/cuda_hook_demo1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/cuda_hook_demos/cuda_hook_demo1 /root/cuda_hook_demos/cuda_hook_demo1/test_app /root/cuda_hook_demos/cuda_hook_demo1/build /root/cuda_hook_demos/cuda_hook_demo1/build/test_app /root/cuda_hook_demos/cuda_hook_demo1/build/test_app/CMakeFiles/test_app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test_app/CMakeFiles/test_app.dir/depend

