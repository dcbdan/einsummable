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
CMAKE_SOURCE_DIR = /home/ubuntu/einsummable

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/einsummable/build

# Include any dependencies generated for this target.
include CMakeFiles/llama_exp02.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/llama_exp02.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/llama_exp02.dir/flags.make

CMakeFiles/llama_exp02.dir/llama/exp02.cc.o: CMakeFiles/llama_exp02.dir/flags.make
CMakeFiles/llama_exp02.dir/llama/exp02.cc.o: ../llama/exp02.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/einsummable/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/llama_exp02.dir/llama/exp02.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/llama_exp02.dir/llama/exp02.cc.o -c /home/ubuntu/einsummable/llama/exp02.cc

CMakeFiles/llama_exp02.dir/llama/exp02.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama_exp02.dir/llama/exp02.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/einsummable/llama/exp02.cc > CMakeFiles/llama_exp02.dir/llama/exp02.cc.i

CMakeFiles/llama_exp02.dir/llama/exp02.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama_exp02.dir/llama/exp02.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/einsummable/llama/exp02.cc -o CMakeFiles/llama_exp02.dir/llama/exp02.cc.s

CMakeFiles/llama_exp02.dir/llama/modules.cc.o: CMakeFiles/llama_exp02.dir/flags.make
CMakeFiles/llama_exp02.dir/llama/modules.cc.o: ../llama/modules.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/einsummable/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/llama_exp02.dir/llama/modules.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/llama_exp02.dir/llama/modules.cc.o -c /home/ubuntu/einsummable/llama/modules.cc

CMakeFiles/llama_exp02.dir/llama/modules.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama_exp02.dir/llama/modules.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/einsummable/llama/modules.cc > CMakeFiles/llama_exp02.dir/llama/modules.cc.i

CMakeFiles/llama_exp02.dir/llama/modules.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama_exp02.dir/llama/modules.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/einsummable/llama/modules.cc -o CMakeFiles/llama_exp02.dir/llama/modules.cc.s

CMakeFiles/llama_exp02.dir/llama/misc.cc.o: CMakeFiles/llama_exp02.dir/flags.make
CMakeFiles/llama_exp02.dir/llama/misc.cc.o: ../llama/misc.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/einsummable/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/llama_exp02.dir/llama/misc.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/llama_exp02.dir/llama/misc.cc.o -c /home/ubuntu/einsummable/llama/misc.cc

CMakeFiles/llama_exp02.dir/llama/misc.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama_exp02.dir/llama/misc.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/einsummable/llama/misc.cc > CMakeFiles/llama_exp02.dir/llama/misc.cc.i

CMakeFiles/llama_exp02.dir/llama/misc.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama_exp02.dir/llama/misc.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/einsummable/llama/misc.cc -o CMakeFiles/llama_exp02.dir/llama/misc.cc.s

CMakeFiles/llama_exp02.dir/llama/builder.cc.o: CMakeFiles/llama_exp02.dir/flags.make
CMakeFiles/llama_exp02.dir/llama/builder.cc.o: ../llama/builder.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/einsummable/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/llama_exp02.dir/llama/builder.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/llama_exp02.dir/llama/builder.cc.o -c /home/ubuntu/einsummable/llama/builder.cc

CMakeFiles/llama_exp02.dir/llama/builder.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama_exp02.dir/llama/builder.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/einsummable/llama/builder.cc > CMakeFiles/llama_exp02.dir/llama/builder.cc.i

CMakeFiles/llama_exp02.dir/llama/builder.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama_exp02.dir/llama/builder.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/einsummable/llama/builder.cc -o CMakeFiles/llama_exp02.dir/llama/builder.cc.s

# Object files for target llama_exp02
llama_exp02_OBJECTS = \
"CMakeFiles/llama_exp02.dir/llama/exp02.cc.o" \
"CMakeFiles/llama_exp02.dir/llama/modules.cc.o" \
"CMakeFiles/llama_exp02.dir/llama/misc.cc.o" \
"CMakeFiles/llama_exp02.dir/llama/builder.cc.o"

# External object files for target llama_exp02
llama_exp02_EXTERNAL_OBJECTS =

llama_exp02: CMakeFiles/llama_exp02.dir/llama/exp02.cc.o
llama_exp02: CMakeFiles/llama_exp02.dir/llama/modules.cc.o
llama_exp02: CMakeFiles/llama_exp02.dir/llama/misc.cc.o
llama_exp02: CMakeFiles/llama_exp02.dir/llama/builder.cc.o
llama_exp02: CMakeFiles/llama_exp02.dir/build.make
llama_exp02: libeinsummable.a
llama_exp02: src/proto/libproto.a
llama_exp02: /usr/lib/x86_64-linux-gnu/libprotobuf.so
llama_exp02: CMakeFiles/llama_exp02.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/einsummable/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable llama_exp02"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/llama_exp02.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/llama_exp02.dir/build: llama_exp02

.PHONY : CMakeFiles/llama_exp02.dir/build

CMakeFiles/llama_exp02.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/llama_exp02.dir/cmake_clean.cmake
.PHONY : CMakeFiles/llama_exp02.dir/clean

CMakeFiles/llama_exp02.dir/depend:
	cd /home/ubuntu/einsummable/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/einsummable /home/ubuntu/einsummable /home/ubuntu/einsummable/build /home/ubuntu/einsummable/build /home/ubuntu/einsummable/build/CMakeFiles/llama_exp02.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/llama_exp02.dir/depend

