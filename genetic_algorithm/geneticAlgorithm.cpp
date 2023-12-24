
#include <genetic_algorithm/genetic_algorithm.hpp>

#include "geneticAlgorithmImpl.hpp"

/* static */ std::unique_ptr<genetic_algorithm> genetic_algorithm::make()
{
    return std::make_unique<CGeneticAlgorithmImpl>();
}

CGeneticAlgorithmImpl::CGeneticAlgorithmImpl()
{

}
