mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=E:\libtorch\Debug;E:\opencv\install-4120\x64\vc17\lib ..
cmake --build . --config Debug
cmake -DCMAKE_PREFIX_PATH=E:\libtorch\Release;E:\opencv\install-4120\x64\vc17\lib ..
cmake --build . --config Release
cd ..

mkdir ..\..\TorchApp\bin
mkdir ..\..\TorchApp\lib
copy /Y build\Debug\*.dll ..\..\TorchApp\bin\
copy /Y build\Release\*.dll ..\..\TorchApp\bin\
copy /Y build\Debug\*.lib ..\..\TorchApp\lib\
copy /Y build\Release\*.lib ..\..\TorchApp\lib\

copy /Y x64\Debug\*.dll ..\..\TorchApp\bin\
copy /Y x64\Release\*.dll ..\..\TorchApp\bin\
copy /Y x64\Debug\*.lib ..\..\TorchApp\lib\
copy /Y x64\Release\*.lib ..\..\TorchApp\lib\
