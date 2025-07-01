# fertool
this is facial expression recognition cli is mainly developed in bunjs/nodejs/typescript with npm moduels of onnxruntime-web and jimp.
## cli flags:
- -i the input folder path where picture files are located in supporting .jpg, jpeg, png, .bmp. by default, $cwd is used.
- -o the output filename which is the recognized results as a csv format file. by default. $cwd/results.csv is used.
- -m the model file path which is the facial expression recognition model file in onnx format. by default, $cwd/model.onnx is used. It also supports FER_MODEL environment variable but it will be overwritten by -m flag.

## Build and Distribution
- The CLI tool will be a binary file compiled and built by bun command
- Supports cross-platform distribution for Windows, macOS with architectures amd64 and arm64
