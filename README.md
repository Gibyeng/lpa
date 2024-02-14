# Large-scale Graph Label Propagation on GPUs

Organization
--------
Code for "Large-scale Graph Label Propagation on GPUs"

Compilation
--------

Requirements

* CMake &gt;= 2.8
* CUDA environment

Compilation is done within the / (root) directory with CMake. 
Please make sure you have installed CMake software compilation tool.
Configure cmakelist.txt appropriately before you start compile. 
To compile, please create a build directory then simple run:

```
mkdir build
cd build
cmake ..
make
```

| Method | Description                                                               |
| ------ | ------------------------------------------------------------------------- |
| CGLP   | Two-kernel compressed label propagation, doing decoding and LP separately |
| NOVA   | Fusing compressing and LP kernels into one kernel                         |


Input arguments
```
./method dataset_path if_compressed if_directed threshold output_file
```
if_compressed is a boolean value decides whether to compress the input graph, if_directed is a boolean value indicts whether the graph is directed or not. Threshold is an integer number controlling nodes whose degree below that value should be compressed.

Examples
```
./nova ../../../dataset/roadNet-CA.txt 1 1 0 result.txt
```
