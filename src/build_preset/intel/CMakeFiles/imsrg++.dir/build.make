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
include CMakeFiles/imsrg++.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/imsrg++.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/imsrg++.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imsrg++.dir/flags.make

CMakeFiles/imsrg++.dir/imsrg++.cc.o: CMakeFiles/imsrg++.dir/flags.make
CMakeFiles/imsrg++.dir/imsrg++.cc.o: ../../imsrg++.cc
CMakeFiles/imsrg++.dir/imsrg++.cc.o: CMakeFiles/imsrg++.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/porro/imsrg/src/build_preset/intel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imsrg++.dir/imsrg++.cc.o"
	/opt/intel/bin/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/imsrg++.dir/imsrg++.cc.o -MF CMakeFiles/imsrg++.dir/imsrg++.cc.o.d -o CMakeFiles/imsrg++.dir/imsrg++.cc.o -c /home/porro/imsrg/src/imsrg++.cc

CMakeFiles/imsrg++.dir/imsrg++.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imsrg++.dir/imsrg++.cc.i"
	/opt/intel/bin/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/porro/imsrg/src/imsrg++.cc > CMakeFiles/imsrg++.dir/imsrg++.cc.i

CMakeFiles/imsrg++.dir/imsrg++.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imsrg++.dir/imsrg++.cc.s"
	/opt/intel/bin/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/porro/imsrg/src/imsrg++.cc -o CMakeFiles/imsrg++.dir/imsrg++.cc.s

# Object files for target imsrg++
imsrg_______OBJECTS = \
"CMakeFiles/imsrg++.dir/imsrg++.cc.o"

# External object files for target imsrg++
imsrg_______EXTERNAL_OBJECTS =

imsrg++: CMakeFiles/imsrg++.dir/imsrg++.cc.o
imsrg++: CMakeFiles/imsrg++.dir/build.make
imsrg++: libVersion.a
imsrg++: libHFMBPT.a
imsrg++: libUnitTest.a
imsrg++: libReferenceImplementations.a
imsrg++: libIMSRGSolver.a
imsrg++: libGenerator.a
imsrg++: libimsrg_util.a
imsrg++: libM0nu.a
imsrg++: libIMSRGSolver.a
imsrg++: libGenerator.a
imsrg++: libimsrg_util.a
imsrg++: libM0nu.a
imsrg++: libReadWrite.a
imsrg++: libVersion.a
imsrg++: libHartreeFock.a
imsrg++: libJacobi3BME.a
imsrg++: libHartreeFock.a
imsrg++: libJacobi3BME.a
imsrg++: libCommutator.a
imsrg++: libCommutator232.a
imsrg++: libCommutator331.a
imsrg++: libDarkMatterNREFT.a
imsrg++: libRPA.a
imsrg++: libOperator.a
imsrg++: libTwoBodyME.a
imsrg++: libThreeBodyME.a
imsrg++: libThreeBodyStorage_iso.a
imsrg++: libThreeBodyStorage_pn.a
imsrg++: libThreeBodyStorage_no2b.a
imsrg++: libThreeBodyStorage_mono.a
imsrg++: boost_src/libIMSRGBoostZip.a
imsrg++: /usr/lib64/libz.so
imsrg++: libThreeBodyStorage.a
imsrg++: libThreeLegME.a
imsrg++: libModelSpace.a
imsrg++: libIMSRGProfiler.a
imsrg++: libAngMomCache.a
imsrg++: /opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so
imsrg++: /opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/libmkl_intel_thread.so
imsrg++: /opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/libmkl_core.so
imsrg++: libPwd.a
imsrg++: /opt/intel/compilers_and_libraries_2020.2.254/linux/compiler/lib/intel64_lin/libiomp5.so
imsrg++: /lib64/libpthread.so
imsrg++: libAngMom.a
imsrg++: /usr/local/lib/libgsl.so
imsrg++: /usr/local/lib/libgslcblas.so
imsrg++: CMakeFiles/imsrg++.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/porro/imsrg/src/build_preset/intel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable imsrg++"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imsrg++.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imsrg++.dir/build: imsrg++
.PHONY : CMakeFiles/imsrg++.dir/build

CMakeFiles/imsrg++.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imsrg++.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imsrg++.dir/clean

CMakeFiles/imsrg++.dir/depend:
	cd /home/porro/imsrg/src/build_preset/intel && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/porro/imsrg/src /home/porro/imsrg/src /home/porro/imsrg/src/build_preset/intel /home/porro/imsrg/src/build_preset/intel /home/porro/imsrg/src/build_preset/intel/CMakeFiles/imsrg++.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imsrg++.dir/depend

