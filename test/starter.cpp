
#include <iostream>

#include <genetic_algorithm/genetic_algorithm.hpp>

bool TestCharSymbol()
{
    auto geneticAlg = genetic_algorithm::make();
    return true;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    return TestCharSymbol() ? 0 : -1;
}