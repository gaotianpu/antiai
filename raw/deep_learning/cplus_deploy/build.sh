# wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.11.0.zip -O /Users/gaotianpu/Downloads/libtorch-macos-1.11.0.zip
# unzip libtorch-macos-1.11.0.zip

rm -fR cmake_install.cmake Makefile CMakeCache.txt CMakeFiles
#mkdir build
#cd build

cp ../example-app.cpp ./example-app.cpp
cp ../CMakeLists.txt ./CMakeLists.txt

cmake -DCMAKE_PREFIX_PATH=/Users/gaotianpu/Downloads/libtorch
cmake --build . --config Release