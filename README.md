# genetic_algorithm
Genetic Algorithm implementation based on OpenCL

# what to do:
1. fix phase 7: it can be done on GPU
1. change data type to half (float 16 bit)
1. change offset type to 32 uint32_t
1. implement for network the ability to work with layer. Specify its inputs, activation function.
1. implement net calculation in parallel inside one layer
1. make the quicket device selection before beginning
1. make all checks for externally obtained settings file
1. settings file: make user friendly settings file output ("external input" / "output" instead of uint64_t)
1. use pointers for json file search data