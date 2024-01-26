
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include <artificial_neural_network/genetic_algorithm.hpp>

#include "geneticAlgorithmImpl.hpp"

using namespace artificial_neural_network;
using namespace std::string_literals;

/* static */ std::unique_ptr<genetic_algorithm> genetic_algorithm::make(const std::unique_ptr<net_structure>& netStructure, test_datas testDatas, const nlohmann::json& settings, bool startCalculationsAutomatically)
{
    return std::make_unique<CGeneticAlgorithmImpl>(netStructure, std::move(testDatas), settings, startCalculationsAutomatically);
}

CGeneticAlgorithmImpl::CGeneticAlgorithmImpl(const std::unique_ptr<net_structure>& netStructure, test_datas testDatas, const nlohmann::json& settings, bool startCalculationsAutomatically) :
    _testDatas(std::move(testDatas)), _settings(settings)
{
    if (settings.contains("calc results"))
        _netStructure = std::make_unique<CNetStructureImpl>(settings.at("calc results").at("data version 1").at("ann structure"));
    else
        _netStructure = std::make_unique<CNetStructureImpl>(netStructure);

    _neurons = _netStructure->_neurons;
    _inputsOffs = _netStructure->_inputsOff;
    _initFuture = std::async(std::launch::async, [this, startCalculationsAutomatically]() { Init(startCalculationsAutomatically); });
    _status = status::algorithm_init;
}

void CGeneticAlgorithmImpl::Init(bool startCalculationsAutomatically)
{

    try
    {
        CheckData();
        CalculateAmounts();
        InitPlatform();
        InitCalculations();

        _status = status::ready_for_calculations;

        if (startCalculationsAutomatically)
            StartCalculations();

    }
    catch (std::runtime_error err)
    {
        _status = status::init_error;
        _lastError = err.what();
    }

}

bool CGeneticAlgorithmImpl::start_calculations() noexcept
{
    if (_status != status::ready_for_calculations)
        return false;

    _status = status::starting_calculations;
    _calcs = std::async(std::launch::async, [this]() { StartCalculations(); });

    return true;
}

void CGeneticAlgorithmImpl::CheckData()
{
    if (_testDatas.empty())
        throw std::runtime_error("Test data cannot be empty");

    for (auto testDataIt = _testDatas.cbegin(); testDataIt != _testDatas.cend(); ++testDataIt)
    {
        if (testDataIt->inputs.empty())
            throw std::runtime_error("Input data has not to be empty");
        if (testDataIt->interpret_data.empty())
            throw std::runtime_error("Interpreting data has not to be empty");
        if (testDataIt->inputs.size() != _netStructure->InputsAmount())
            throw std::runtime_error("Inputs size in test data are not equal to the network inputs amount");
        if (testDataIt->interpret_data.size() != _testDatas.front().interpret_data.size())
            throw std::runtime_error("Interpreting data is not the same for all testing set");
    }
}

void CGeneticAlgorithmImpl::CalculateAmounts()
{
    const nlohmann::json& generatedNets = _settings["generated nets"];
    _randomRange = generatedNets["random value +- range"];
    _initialNets = generatedNets["initial amount"];
    _initialRandoms = static_cast<size_t>(generatedNets["initial random"].get<double>() * _initialNets);
    _initialWorst = static_cast<size_t>(generatedNets["initial worst"].get<double>() * _initialNets);

    if (_initialRandoms + _initialWorst >= _initialNets)
        throw std::runtime_error("Initial nets amount "s + std::to_string(_initialNets) + " must be bigger than sum of random "s + std::to_string(_initialRandoms) + "and initial worst "s + std::to_string(_initialWorst) + " nets");

    _initialBest = _initialNets - (_initialRandoms + _initialWorst);
    _initialStageAmount = _initialNets;

    _crossingoverBrothers = generatedNets["crossingover brothers"];
    _crossingoverStageAmount = _initialStageAmount * ( _initialStageAmount - 1) / 2 * _crossingoverBrothers;

    _mutations.append_range(generatedNets["mutations"]);
    _mutationsStageAmount = (_initialStageAmount + _crossingoverStageAmount) * _mutations.size();

    _generalNetsAmount = _initialStageAmount + _crossingoverStageAmount + _mutationsStageAmount;

    _configsAmount = _netStructure->ConfigsAmount();
    _netConfigsBytes = _configsAmount;
    _allNetsConfigsBytes = _generalNetsAmount * _netConfigsBytes;

    _inputsPerNet = _netStructure->InputsAmount();
    _outputsPerNet = _netStructure->OutputsAmount();
}

void CGeneticAlgorithmImpl::InitPlatform()
{
//    std::vector<cl::Platform> platforms;
//    cl::Platform::get(&platforms);
//
//    std::vector<cl::Device> devices;
//
//    for (const cl::Platform& platform : platforms)
//    {
//        std::vector<cl::Device> platformDevices;
//        platform.getDevices(CL_DEVICE_TYPE_GPU, &platformDevices);
//        devices.append_range(platformDevices);
//    }
//
//    std::sort(devices.begin(), devices.end(), [](const cl::Device& a, const cl::Device& b)
//    {
//        cl_uint frequencyA;
//        cl_uint frequencyB;
//
//        a.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &frequencyA);
//        b.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &frequencyB);
//
//        return frequencyA > frequencyB;
//    });
//
//#ifdef _DEBUG
//    if (devices.size() > 1)
//    {
//        cl_uint frequencyFirst;
//        cl_uint frequencySecond;
//
//        devices.at(0).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &frequencyFirst);
//        devices.at(1).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &frequencySecond);
//
//        assert(frequencyFirst >= frequencySecond);
//    }
//#endif // _DEBUG
    // Filter for a 2.0 or newer platform and set it as the default
//    {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform;
    for (auto &p : platforms)
    {
        std::string platformVer = p.getInfo<CL_PLATFORM_VERSION>();
        if (platformVer.find("OpenCL 2.") != std::string::npos ||
            platformVer.find("OpenCL 3.") != std::string::npos)
        {
            platform = p;
        }
    }
    if (platform() == 0)
        throw std::runtime_error("No OpenCL 2.0 or newer platform found");

    _platform = cl::Platform::setDefault(platform);
    if (_platform != platform)
        throw std::runtime_error("Error setting default platform");

    try
    {
        std::vector<std::string> srcCode;
        
        srcCode.emplace_back(_definesProg);

        if (!_settings.contains("fitness function calculation"))
            throw std::runtime_error("There is no fitness function calculation information in \"fitness function calculation\" settings branch");

        const nlohmann::json& fitnessCalcSettings = _settings["fitness function calculation"];
        if (!fitnessCalcSettings.contains("source code"))
            throw std::runtime_error("There is no fitness function calculation source code in \"fitness function calculation/source code\" settings branch");
        if (!fitnessCalcSettings.contains("function name"))
            throw std::runtime_error("There is no fitness function calculation kernel function name in \"fitness function calculation/function name\" settings branch");

        _fitnessFunctionCalcName = fitnessCalcSettings["function name"];
        srcCode.emplace_back("TData TestCheck"s + fitnessCalcSettings["source code"].get<std::string>());
        
        srcCode.emplace_back(_randomProg);
        srcCode.emplace_back(_geneticAlgProg);
        
        _clProg = std::make_unique<decltype(_clProg)::element_type>(srcCode);

        std::string buildOptions = "-cl-std=CL3.0";
        if constexpr (_testGeneticAlgorithm)
            buildOptions += " -D TEST_GENETIC_ALGORITHM="s + std::to_string(_testGeneticAlgorithm);
        _clProg->build(buildOptions.c_str());
    }
    catch (...)
    {
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = _clProg->getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        std::string errDescr;
        for (auto& pair : buildInfo)
            errDescr += (errDescr.empty() ? ""s : ", "s) + pair.second;

        assert(false);
        throw std::runtime_error("Error while building kernel code"s + (errDescr.empty() ? ""s : "\n"s + errDescr));
    }

    _phase1Call_Randomizing      = std::make_unique<decltype(_phase1Call_Randomizing)::element_type>(     *_clProg.get(), "Phase1_Randomizing"     );
    _phase1Call_RandomizeAll     = std::make_unique<decltype(_phase1Call_RandomizeAll)::element_type>(    *_clProg.get(), "Phase1_RandomizeAll"    );
    _phase2Call_Crossingover     = std::make_unique<decltype(_phase2Call_Crossingover)::element_type>(    *_clProg.get(), "Phase2_Crossingover"    );
    _phase3Call_PrepareMutations = std::make_unique<decltype(_phase3Call_PrepareMutations)::element_type>(*_clProg.get(), "Phase3_PrepareMutations");
    _phase4Call_Mutations        = std::make_unique<decltype(_phase4Call_Mutations)::element_type>(       *_clProg.get(), "Phase4_Mutations"       );
    _phase5Call_NetsCalculations = std::make_unique<decltype(_phase5Call_NetsCalculations)::element_type>(*_clProg.get(), "Phase5_NetsCalculations");
    _phase6Call_CalcFitnessFunct = std::make_unique<decltype(_phase6Call_CalcFitnessFunct)::element_type>(*_clProg.get(), "Phase6_CalcFitnessFunct");
}

void CGeneticAlgorithmImpl::InitCalculations()
{
    _netConfigs.resize(_configsAmount * _generalNetsAmount);
    std::fill(_netConfigs.begin(), _netConfigs.end(), 0);
    InitBuffer(_netConfigs, _netConfigsCl, false, false);

    _allRandomNets = true;
    _toStop = false;

    _netsToRandomize.resize(_initialNets);
    std::fill(_netsToRandomize.begin(), _netsToRandomize.end(), true);
    InitBuffer(_netsToRandomize, _netsToRandomizeCl, false, false);

    _crossingoverParent1.resize(_crossingoverStageAmount / _crossingoverBrothers);
    _crossingoverParent2.resize(_crossingoverStageAmount / _crossingoverBrothers);
    size_t off = 0;
    for (TOffset parent1 = 0; parent1 < _initialNets; ++parent1)
        for (TOffset parent2 = parent1 + 1; parent2 < _initialNets; ++parent2)
        {
            _crossingoverParent1[off] = parent1;
            _crossingoverParent2[off] = parent2;
            ++off;
        }

    InitBuffer(_crossingoverParent1, _crossingoverParent1Cl, true, false);
    InitBuffer(_crossingoverParent2, _crossingoverParent2Cl, true, false);

    cl::copy(_crossingoverParent1.begin(), _crossingoverParent1.end(), _crossingoverParent1Cl);
    cl::copy(_crossingoverParent2.begin(), _crossingoverParent2.end(), _crossingoverParent2Cl);

    _mutationsAmounts.resize(_mutations.size());
    for (size_t i = 0; i < _mutations.size(); ++i)
        _mutationsAmounts[i] = std::max(static_cast<TOffset>(1), static_cast<TOffset>(_mutations[i] * _configsAmount));

    InitBuffer(_mutationsAmounts, _mutationsAmountsCl, true, false);
    cl::copy(_mutationsAmounts.begin(), _mutationsAmounts.end(), _mutationsAmountsCl);

    _maxMutationsAmount = _mutationsAmounts[0];
    for (auto mutationsAmount : _mutationsAmounts)
        _maxMutationsAmount = (std::max)(_maxMutationsAmount, mutationsAmount);

    _phase1RandomizingThreads = _configsAmount * _initialNets;
    _phase1RandomizeAllThreads = _configsAmount * _generalNetsAmount;
    _phase2Threads = _configsAmount * _crossingoverStageAmount;
    _phase3Threads = _configsAmount * (_initialNets + _crossingoverStageAmount) * _mutations.size();
    _phase4Threads = _mutationsStageAmount * _mutations.size() * _maxMutationsAmount;
    _phase5Threads = _generalNetsAmount * _testDatas.size();
    _phase6Threads = _generalNetsAmount;

    InitRandom();
    InitNeuralNetworkStructure();

    _fitnessFunction.resize(_generalNetsAmount);
    InitBuffer(_fitnessFunction, _fitnessFunctionCl, true, false);

    _fitnessFunctOrder.resize(_generalNetsAmount);

    const nlohmann::json& fitnessCalcSettings = _settings["fitness function calculation"];
    _rejectFitnessFunctionLevel = fitnessCalcSettings["reject level"];
    _saveFitnessFunctionLevel = fitnessCalcSettings["save level"];
    _stopFitnessFunctionLevel = fitnessCalcSettings["stop level"];

    if (_settings.contains("calc results"))
        InitState();
}

void CGeneticAlgorithmImpl::InitNeuralNetworkStructure()
{
    _inputs.resize(_inputsPerNet * _testDatas.size());
    InitBuffer(_inputs, _inputsCl, true, false);

    _testDataInfos.resize(_testDatas.size() * _testDatas.at(0).interpret_data.size());
    InitBuffer(_testDataInfos, _testDataInfosCl, true, false);

    auto pInput = _inputs.begin();
    auto pTestData = _testDataInfos.begin();
    for (const test_data& testData : _testDatas)
    {
        std::copy(testData.inputs.begin(), testData.inputs.end(), pInput);
        pInput += testData.inputs.size();

        std::copy(testData.interpret_data.begin(), testData.interpret_data.end(), pTestData);
        pTestData += testData.interpret_data.size();
    }
    assert(pInput == _inputs.end());
    assert(pTestData == _testDataInfos.end());

    cl::copy(_inputs.begin(), _inputs.end(), _inputsCl);
    cl::copy(_testDataInfos.begin(), _testDataInfos.end(), _testDataInfosCl);

    _outputs.resize(_outputsPerNet * _testDatas.size() * _generalNetsAmount);
    InitBuffer(_outputs, _outputsCl, true, false);

    InitBuffer(_neurons, _neuronsCl, true, false);
    cl::copy(_neurons.begin(), _neurons.end(), _neuronsCl);

    InitBuffer(_inputsOffs, _inputsOffsCl, true, false);
    cl::copy(_inputsOffs.begin(), _inputsOffs.end(), _inputsOffsCl);

    _neuronsStates.resize(_netStructure->_statesSize * _generalNetsAmount * _testDatas.size());
    InitBuffer(_neuronsStates, _neuronsStatesCl, false, false);
}

void CGeneticAlgorithmImpl::InitRandom()
{
    size_t randomElemsSize = std::max({ _phase1RandomizingThreads, _phase1RandomizeAllThreads, _phase2Threads, _phase4Threads });
    _randomBufferState.resize(randomElemsSize);
    InitBuffer(_randomBufferState, _randomBufferStateCl, false, false);

    std::vector<size_t> randomBufferSeed(randomElemsSize);
    cl::Buffer randomBufferSeedCl;
    InitBuffer(randomBufferSeed, randomBufferSeedCl, false, false);

    size_t rnd_init = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 generator(rnd_init);
    for (size_t& value : randomBufferSeed)
        value = generator();

    cl::copy(randomBufferSeed.begin(), randomBufferSeed.end(), randomBufferSeedCl);

    cl::KernelFunctor<cl::Buffer, cl::Buffer> randomBufferInit(*_clProg.get(), "Init_Random");
    randomBufferInit( cl::EnqueueArgs(cl::NDRange(randomElemsSize)), _randomBufferStateCl, randomBufferSeedCl);

    if constexpr (_testGeneticAlgorithm == 1)
    {
        std::vector<size_t> randomBufferSizeT(randomElemsSize);
        cl::Buffer randomBufferSizeTCl;
        InitBuffer(randomBufferSizeT, randomBufferSizeTCl, false, true);

        std::vector<double> randomBufferDouble(randomElemsSize);
        cl::Buffer randomBufferDoubleCl;
        InitBuffer(randomBufferDouble, randomBufferDoubleCl, false, true);

        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> randomBufferTest(*_clProg.get(), "Test_Random");
        randomBufferTest( cl::EnqueueArgs(cl::NDRange(randomElemsSize)), _randomBufferStateCl, randomBufferSizeTCl, randomBufferDoubleCl);

        cl::copy(randomBufferSizeTCl, randomBufferSizeT.begin(), randomBufferSizeT.end());
        cl::copy(randomBufferDoubleCl, randomBufferDouble.begin(), randomBufferDouble.end());

        std::set<size_t> randomBufferSetSizeT(randomBufferSizeT.begin(), randomBufferSizeT.end());
        std::set<double> randomBufferSetDouble(randomBufferDouble.begin(), randomBufferDouble.end());

        assert(randomBufferSetSizeT.size()  > randomElemsSize * 0.75);
        assert(randomBufferSizeT[2] - randomBufferSizeT[1] != randomBufferSizeT[1] - randomBufferSizeT[0]);
        assert(randomBufferSetDouble.size() > randomElemsSize * 0.75);
    }

}

size_t CGeneticAlgorithmImpl::GlobalMemoryBytesRequirement()
{
    size_t bytes = 0;

//    bytes += sizeof(data_type) * _testDatas.size() * (_testDatas.front().inputs.size() + _testDatas.front().interpret_data.size());
    
//    bytes += sizeof(artificial_neural_network::CNetStructureImpl::TOffset) * (_netStructure->InputsAmount() + _netStructure->OutputsAmount());
    return bytes;
}

template<typename T>
void CGeneticAlgorithmImpl::InitBuffer(T& vector, cl::Buffer& buffer, size_t size, bool readOnly, bool useHostPtr)
{
    vector.resize(size);
    InitBuffer(vector, buffer, readOnly, useHostPtr);
}

template<typename T>
void CGeneticAlgorithmImpl::InitBuffer(T& vector, cl::Buffer& buffer, bool readOnly, bool useHostPtr)
{
    assert(!vector.empty());
    buffer = cl::Buffer(vector.begin(), vector.end(), readOnly, useHostPtr);
}

nlohmann::json CGeneticAlgorithmImpl::export_calculation_result() noexcept
{
    nlohmann::json generalResult;
    nlohmann::json& result = generalResult["data version 1"];

    result["ann structure"] = _netStructure->Export();

    {
        std::lock_guard<std::mutex> configsToBeExportedLock(_configsToBeExportedMutex);
        for (const TDatas& configs : _configsToBeExported)
            result["configs"].emplace_back(nlohmann::json(configs));
    }

    return generalResult;
}

bool CGeneticAlgorithmImpl::has_calculation_result() noexcept
{
    std::lock_guard<std::mutex> configsToBeExportedLock(_configsToBeExportedMutex);

    return !_configsToBeExported.empty();
}

void CGeneticAlgorithmImpl::InitState()
{
    if (!_settings.contains("calc results"))
        return;

    auto dst = _netConfigs.begin();
    size_t i = 0;
    for (const nlohmann::json& config : _settings.at("calc results").at("data version 1").at("configs"))
    {
        _netsToRandomize[i] = false;

        assert(config.size() == _configsAmount);
        for (const auto value : config)
            *dst++ = value;
        ++i;
    }

    cl::copy(_netsToRandomize.cbegin(), _netsToRandomize.cend(), _netsToRandomizeCl);
    cl::copy(_netConfigs.cbegin(), _netConfigs.cend(), _netConfigsCl);

    _allRandomNets = false;
}
