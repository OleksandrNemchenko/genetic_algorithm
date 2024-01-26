
#ifndef _AI_GENETIC_ALGORITHM_HPP
#define _AI_GENETIC_ALGORITHM_HPP

#include <memory>
#include <vector>

#include <artificial_neural_network/utilities.hpp>
#include <nlohmann/json.hpp>

namespace artificial_neural_network
{

class net_structure;

class genetic_algorithm
{
public:
    enum status {algorithm_init, ready_for_work, init_error, ready_for_calculations, calc_error, starting_calculations, calc_phase1, calc_phase2, calc_phase3, calc_phase4, calc_phase5, calc_phase6, calc_phase7, calc_phase8, got_result, unknown};
    
    struct test_data
    {
        datas_type inputs;
        datas_type interpret_data;
    };

    using test_datas = std::vector<test_data>;

    static std::unique_ptr<genetic_algorithm> make(const std::unique_ptr<net_structure>& netStructure, test_datas testDatas, const nlohmann::json& settings, bool startCalculationsAutomatically = true);
    virtual ~genetic_algorithm() noexcept = default;

    virtual bool start_calculations() noexcept = 0;
    virtual void stop_calculations() noexcept = 0;
    virtual nlohmann::json export_calculation_result() noexcept = 0;
    virtual bool has_calculation_result() noexcept = 0;

    virtual status current_status() const noexcept = 0;
    virtual const std::string& last_error() const noexcept = 0;
    virtual fitness_funct_type best_fitness_function() const noexcept = 0;
    virtual size_t iteration() const noexcept = 0;

};

}   // namespace artificial_neural_network

#endif // _AI_GENETIC_ALGORITHM_HPP
