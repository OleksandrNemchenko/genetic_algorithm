
#include <bitset>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>

#include <artificial_neural_network/net_structure.hpp>
#include <artificial_neural_network/genetic_algorithm.hpp>
#include <nlohmann/json.hpp>

using namespace artificial_neural_network;

bool TestCharSymbol(const std::filesystem::path& genAlgCalcFile)
{
    constexpr size_t bits = 8;
    constexpr size_t outsAmount = 2;
    auto annStr = net_structure::Make(bits, outsAmount);
    std::set<char> ukrAll;
    std::set<char> ukrBig;

    for (char smb : std::string("абвгґдеєжзіїйклмнопрстуфхцчшщьюя"))
        ukrAll.emplace(smb);

    for (char smb : std::string("АБВГҐДЕЄЖЗІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"))
    {
        ukrAll.emplace(smb);
        ukrBig.emplace(smb);
    }

    using af = artificial_neural_network::net_structure::EActivationFunction;
    annStr->AddP2PNeuronsLayer();
    annStr->AddFullyConnectedNeuronsLayer();
    annStr->AddFullyConnectedNeuronsLayer();
    annStr->AddOutputLayer();

    genetic_algorithm::test_datas testDatas;
    for (int smb = 0; smb <= 255; ++smb)
    {
        genetic_algorithm::test_data testData;

        std::bitset<bits> input = smb;
        for (int bit = 0; bit < bits; ++bit)
            testData.inputs.emplace_back(input[bit]);
        testData.interpret_data.emplace_back(static_cast<uint8_t>(ukrAll.contains(static_cast<char>(smb)) ? 1 : 0));
        testData.interpret_data.emplace_back(static_cast<uint8_t>(ukrBig.contains(static_cast<char>(smb)) ? 1 : 0));

        testDatas.emplace_back(std::move(testData));
    }
    static const size_t testDatasAmount = testDatas.size();

    nlohmann::json settings;

    if (std::filesystem::exists(genAlgCalcFile))
    {
        std::ifstream genAlgCalc(genAlgCalcFile);
        std::string genAlgCalcStr{ std::istreambuf_iterator<char>(genAlgCalc), std::istreambuf_iterator<char>() };
        if (genAlgCalc.is_open())
            settings["calc results"] = nlohmann::json::parse(genAlgCalcStr);

        std::cout << "- providing previous calculation results from file " << genAlgCalcFile << std::endl;
    }

    nlohmann::json& generatedNets = settings["generated nets"];
    generatedNets["random value +- range"] = 2;
    generatedNets["initial amount"] = 30;
    generatedNets["initial random"] = 0.5;
    generatedNets["initial worst"] = 0.1;
    generatedNets["crossingover brothers"] = 10;
    generatedNets["mutations"] = { 0.0001, 0.001, 0.01, 0.1, 0.9 };

    nlohmann::json& fitnessFunction = settings["fitness function calculation"];
    fitnessFunction["function name"] = "TestCheck";
    fitnessFunction["reject level"] = testDatasAmount / 2;
    fitnessFunction["save level"] = testDatasAmount * 2;
    fitnessFunction["stop level"] = testDatasAmount * 2 + ukrAll.size();
    fitnessFunction["source code"] = R"(
(__global const TData* outputs, __global const TData* testData, const TOffset datasetsAmount)
{
    TData sum = 0;
    for (TOffset i = 0; i < datasetsAmount; ++i, outputs += 2, testData += 2)
    {
        const bool expectedIsUkr = testData[0] == 1;
        const bool actualIsUkr   = outputs[0] > 0.5;

        const bool expectedIsUkrBig = testData[1] == 1;
        const bool actualIsUkrBig   = outputs[1] > 0.5;

        sum += (expectedIsUkr == actualIsUkr ? 2 : 0) + (actualIsUkr && expectedIsUkrBig == actualIsUkrBig ? 1 : 0);
    }

    return sum;
};)";

/*
    auto annStr = net_structure::Make(3, 2);
    annStr->AddP2PNeuronsLayer();
    annStr->AddFullyConnectedNeuronsLayer(2, artificial_neural_network::net_structure::ESourceType::NEURONS);
    annStr->AddOutputLayer(artificial_neural_network::net_structure::EConnectionType::FULLY_CONNECTED);

    genetic_algorithm::test_datas testDatas;
     
    {
        genetic_algorithm::test_data testData;
        testData.inputs = { 1, 1, 2 };
        testData.interpret_data = {1,1};
        testDatas.emplace_back(std::move(testData));
    }
     
    {
        genetic_algorithm::test_data testData;
        testData.inputs = { 5, 2, 7 };
        testData.interpret_data = {2,10};
        testDatas.emplace_back(std::move(testData));
    }
     
    {
        genetic_algorithm::test_data testData;
        testData.inputs = { 8, -10, 6 };
        testData.interpret_data = {0.5,0};
        testDatas.emplace_back(std::move(testData));
    }

    nlohmann::json settings;

    nlohmann::json& generatedNets = settings["generated nets"];
    generatedNets["random value +- range"] = 5;
    generatedNets["initial amount"] = 10;
    generatedNets["initial random"] = 0.5;
    generatedNets["initial worst"] = 0.1;
    generatedNets["crossingover brothers"] = 5;
    generatedNets["mutations"] = { 0.001, 0.1, 0.9 };

    nlohmann::json& fitnessFunction = settings["fitness function calculation"];
    fitnessFunction["function name"] = "TestCheck";
    fitnessFunction["reject level"] = 0.0;
    fitnessFunction["stop level"] = 1000.0;
    fitnessFunction["source code"] = R"(
(__global const TData* outputs, __global const TData* testData, const TOffset datasetsAmount)
{
    TData sum = 0;
    for (TOffset i = 0; i < datasetsAmount; ++i, outputs += 2, testData += 2)
        sum += (outputs[0] + outputs[1]) * testData[0] - testData[1];
    return sum;
};)";
*/
    auto geneticAlg = genetic_algorithm::make(annStr, testDatas, settings);
    artificial_neural_network::fitness_funct_type lastFitnessFunction = std::numeric_limits<artificial_neural_network::fitness_funct_type>::min();
    bool firstCalcResultsFileSaving = true;

    std::cout << "- initializing... " << std::endl;

    std::chrono::steady_clock::time_point previousReport;
    static const std::chrono::seconds reportTimeout{ 30 };

    auto updateReportTimepoint = [&previousReport]()
    {
        previousReport = std::chrono::steady_clock::now();
    };
    updateReportTimepoint();

    while (geneticAlg->current_status() != genetic_algorithm::status::got_result)
    {
        if (geneticAlg->current_status() == genetic_algorithm::status::init_error || geneticAlg->current_status() == genetic_algorithm::status::calc_error)
        {
            std::cout << "* error: " << geneticAlg->last_error() << std::endl;
            break;
        }

        if (std::chrono::steady_clock::now() - previousReport > reportTimeout || 
            geneticAlg->current_status() != genetic_algorithm::status::algorithm_init && lastFitnessFunction != geneticAlg->best_fitness_function())
        {
            lastFitnessFunction = geneticAlg->best_fitness_function();
            std::cout << "- fitness value (ideal / best / reject) = " << fitnessFunction["stop level"].get<artificial_neural_network::fitness_funct_type>() << " / " << lastFitnessFunction << " / " << fitnessFunction["reject level"].get<artificial_neural_network::fitness_funct_type>() << ", iteration " << geneticAlg->iteration() << std::endl;

            if (geneticAlg->has_calculation_result())
            {
                std::ofstream exportResultFile(genAlgCalcFile);
                assert(exportResultFile.is_open());
                exportResultFile << geneticAlg->export_calculation_result().dump(4);
            }

            if (firstCalcResultsFileSaving)
            {
                firstCalcResultsFileSaving = false;
                std::cout << "- start saving calculation results to file " << genAlgCalcFile << std::endl;
            }

            updateReportTimepoint();

        }

        std::this_thread::sleep_for(std::chrono::milliseconds{ 333 });
    }
    
    if (geneticAlg->current_status() == genetic_algorithm::status::got_result)
        std::cout << "- got result: best fitness value = " << lastFitnessFunction << ", iterations " << geneticAlg->iteration() << std::endl;

    return true;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    return TestCharSymbol(argc == 1 ? std::filesystem::temp_directory_path() / "export_genetic_alg_net_calc.json" : argv[1]) ? 0 : -1;
}
