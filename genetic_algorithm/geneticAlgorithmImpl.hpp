
#ifndef _AI_GENETIC_ALGORITHM_IMPL_HPP
#define _AI_GENETIC_ALGORITHM_IMPL_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY

#include <CL/opencl.hpp>

#include <genetic_algorithm/genetic_algorithm.hpp>

class CGeneticAlgorithmImpl : public genetic_algorithm
{
public:
    ~CGeneticAlgorithmImpl() noexcept override = default;

private:
};

#endif // _AI_GENETIC_ALGORITHM_IMPL_HPP
