mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=E:\libtorch\Debug;E:\opencv\install-4120\x64\vc17\lib ..
cmake --build . --config Debug
cmake -DCMAKE_PREFIX_PATH=E:\libtorch\Release;E:\opencv\install-4120\x64\vc17\lib ..
cmake --build . --config Release
cd ..