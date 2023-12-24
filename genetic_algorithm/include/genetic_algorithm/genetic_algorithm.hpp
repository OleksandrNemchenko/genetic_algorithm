
#ifndef _AI_GENETIC_ALGORITHM_HPP
#define _AI_GENETIC_ALGORITHM_HPP

#include <memory>

class genetic_algorithm
{
public:
    static std::unique_ptr<genetic_algorithm> make();
    virtual ~genetic_algorithm() noexcept = default;

};

#endif // _AI_GENETIC_ALGORITHM_HPP
