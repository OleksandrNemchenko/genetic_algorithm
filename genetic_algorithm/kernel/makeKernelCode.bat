python makeKernelCode.py ^
	--source_cpp_pattern_file="kernelCodePattern.cpp" ^
	--source_cpp_pattern="GENETIC_ALG" ^
	--source_c_kernel_file="geneticAlgSrc.cpp" ^
	--source_c_kernel_prefix="/* KERNEL CODE BEGINNING */" ^
	--source_c_kernel_postfix="/* KERNEL CODE ENDING */" ^
	--destination_cpp_file="kernelCode.cpp" ^
	--silence

python makeKernelCode.py ^
	--source_cpp_pattern_file="kernelCode.cpp" ^
	--source_cpp_pattern="DEFINES" ^
	--source_c_kernel_file="definesSrc.cpp" ^
	--source_c_kernel_prefix="/* KERNEL CODE BEGINNING */" ^
	--source_c_kernel_postfix="/* KERNEL CODE ENDING */" ^
	--destination_cpp_file="kernelCode.cpp" ^
	--silence

python makeKernelCode.py ^
	--source_cpp_pattern_file="kernelCode.cpp" ^
	--source_cpp_pattern="RANDOM_PROG" ^
	--source_c_kernel_file="tinyMTSrc.cpp" ^
	--source_c_kernel_prefix="/* KERNEL CODE BEGINNING */" ^
	--source_c_kernel_postfix="/* KERNEL CODE ENDING */" ^
	--destination_cpp_file="kernelCode.cpp" ^
	--silence

