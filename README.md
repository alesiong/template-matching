# Template Matching

## How to Run

This program should be run with 3 arguments:

```
template-matching original.bmp template.bmp out.bmp
```

The 3 arguments correspond to the path for input image, template image and output image.

There are `template-matching.exe` and `template-matching-cpu.exe` in the `bin` directory, which are GPU and CPU version built for Windows x64.

### Compiling

#### Windows

1. Open `template-matching.sln` with VS 2015
2. Compile & Run

#### Linux

```
nvcc -O3 -g -o template-matching bmp_util.c kernel.cu main.cu utils.cu --compiler-options -Wall,-Wextra,-Wno-unused-result
```

#### CPU Version

```
gcc -o template-matching-cpu cpumain.c bmp_util.c
```



## File Structure

`bin/`: compiled binary executable

`images/`: images and templates for testing

`includes/`: headers

`bmp_util.c`: functions to read and write BMP

`kernel.cu`: main logic and kernels

`main.cu`: `main` function and starting code

`utils.cu`: some CUDA helper functions

`template-matching.sln`: solution file for VS 2015

`template-matching.vcxproj`: project file for VS 2015

`cpumain.c`: code for CPU matching

## Group Members

### QIU Yuheng 邱渝桁

ID: 115010216

1. Modify kernels to support for images larger than 1024x1024
2. Write the `main` function



### YU Pengfei 余鹏飞

ID: 115010271

1. Write CPU code for preprocessing (include streams)
2. Write the main logic for finding matchings
3. Debug kernels



### YU Xianggang 余湘港

ID: 115010273

1. Write the kernels

