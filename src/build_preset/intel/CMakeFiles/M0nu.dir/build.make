# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/porro/imsrg/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/porro/imsrg/src/build_preset/intel

# Include any dependencies generated for this target.
include CMakeFiles/M0nu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/M0nu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/M0nu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/M0nu.dir/flags.make

CMakeFiles/M0nu.dir/M0nu.cc.o: CMakeFiles/M0nu.dir/flags.make
CMakeFiles/M0nu.dir/M0nu.cc.o: ../../M0nu.cc
CMakeFiles/M0nu.dir/M0nu.cc.o: CMakeFiles/M0nu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/porro/imsrg/src/build_preset/intel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/M0nu.dir/M0nu.cc.o"
	/opt/intel/bin/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/M0nu.dir/M0nu.cc.o -MF CMakeFiles/M0nu.dir/M0nu.cc.o.d -o CMakeFiles/M0nu.dir/M0nu.cc.o -c /home/porro/imsrg/src/M0nu.cc

CMakeFiles/M0nu.dir/M0nu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/M0nu.dir/M0nu.cc.i"
	/opt/intel/bin/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/porro/imsrg/src/M0nu.cc > CMakeFiles/M0nu.dir/M0nu.cc.i

CMakeFiles/M0nu.dir/M0nu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/M0nu.dir/M0nu.cc.s"
	/opt/intel/bin/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/porro/imsrg/src/M0nu.cc -o CMakeFiles/M0nu.dir/M0nu.cc.s

# Object files for target M0nu
M0nu_OBJECTS = \
"CMakeFiles/M0nu.dir/M0nu.cc.o"

# External object files for target M0nu
M0nu_EXTERNAL_OBJECTS =

libM0nu.a: CMakeFiles/M0nu.dir/M0nu.cc.o
libM0nu.a: CMakeFiles/M0nu.dir/build.make
libM0nu.a: CMakeFiles/M0nu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/porro/imsrg/src/build_preset/intel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libM0nu.a"
	$(CMAKE_COMMAND) -P CMakeFiles/M0nu.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/M0nu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/M0nu.dir/build: libM0nu.a
.PHONY : CMakeFiles/M0nu.dir/build

CMakeFiles/M0nu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/M0nu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/M0nu.dir/clean

CMakeFiles/M0nu.dir/depend:
	cd /home/porro/imsrg/src/build_preset/intel && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/porro/imsrg/src /home/porro/imsrg/src /home/porro/imsrg/src/build_preset/intel /home/porro/imsrg/src/build_preset/intel /home/porro/imsrg/src/build_preset/intel/CMakeFiles/M0nu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/M0nu.dir/depend

