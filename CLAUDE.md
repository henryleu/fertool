# fertool
this is facial expression recognition cli is mainly developed in bunjs/nodejs/typescript with npm moduels of onnxruntime-web and jimp.
## Cli flags:
- -i the input folder path where picture files are located in supporting .jpg, jpeg, png, .bmp. by default, $cwd is used.
- -o the output filename which is the recognized results as a csv format file. by default. $cwd/results.csv is used.
- -m the model file path which is the facial expression recognition model file in onnx format. by default, $cwd/model.onnx is used. It also supports FER_MODEL environment variable but it will be overwritten by -m flag.

## Terms
- fer: facial expression recognition

## Features
- Supported picture file types: *.jpg, *.jpeg, *.png, *.bmp, *.tiff, *.webp
- Supported model specs:
  - input shape: 1*3*320*320
  - output shape: 1*10
  - output labels: angry, disgust, fear, happy, neutral, sad
- fer a picture folder and output the results as a csv file
  - list and filter picture files in input folder order by filename asc (Alphabetically). such as: 001.jpg, 002.jpg, 003.png, etc.
  - fer a picture file and cache the result one by one
  - output all results to a csv file.
  - each picture file will be recognized as a csv line in the output file.
  - each line of the output file will be: picture filename, recognized result.
  - recognized result is like: 001.png,  fear/60.12%, happy/30.2%, sad/6.6%, angry/0.5%, disgust/1.2%, neutral/9.1%

## Build and Distribution
- The CLI tool will be a binary file compiled and built by bun command
- Supports cross-platform distribution for Windows, macOS with architectures amd64 and arm64
