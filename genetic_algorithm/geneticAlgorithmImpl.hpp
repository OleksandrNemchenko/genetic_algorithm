
#ifndef _AI_GENETIC_ALGORITHM_IMPL_HPP
#define _AI_GENETIC_ALGORITHM_IMPL_HPP

#include <memory>
#include <mutex>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY

#include <atomic>
#include <future>

#include <CL/opencl.hpp>

#include <artificial_neural_network/net_structure.hpp>
#include <artificial_neural_network/genetic_algorithm.hpp>
#include <artificial_neural_network/utilities.hpp>
#include <netStructureImpl.hpp>

namespace artificial_neural_network
{

class CGeneticAlgorithmImpl : public genetic_algorithm
{
public:
    CGeneticAlgorithmImpl(const std::unique_ptr<net_structure>& netStructure, test_datas testDatas, const nlohmann::json& settings, bool startCalculationsAutomatically);
    ~CGeneticAlgorithmImpl() noexcept override = default;

    bool start_calculations() noexcept override;
    void stop_calculations() noexcept override { _toStop = true; }
    nlohmann::json export_calculation_result() noexcept override;
    bool has_calculation_result() noexcept override;

    status current_status() const noexcept override { return _status; }
    const std::string& last_error() const noexcept override { return _lastError; }
    fitness_funct_type best_fitness_function() const noexcept override { return _bestFitnessFunction; }
    size_t iteration() const noexcept override { return _iteration; }

private:
    using TData = data_type;
    using TDatas = datas_type;
    using TOffset = offset_type;
    using TOffsets = offsets_type;
    using TFitnessFunctOrder = std::pair<TOffset, TData>;

    static const std::string _definesProg;
    static const std::string _geneticAlgProg;
    static const std::string _randomProg;

    static constexpr size_t _testGeneticAlgorithm =
#ifdef TEST_GENETIC_ALGORITHM
        TEST_GENETIC_ALGORITHM;
#else // TEST_GENETIC_ALGORITHM
        0;
#endif // TEST_GENETIC_ALGORITHM

    std::unique_ptr<CNetStructureImpl> _netStructure;
    const test_datas _testDatas;
    const nlohmann::json& _settings;
    std::atomic<status> _status;
    std::atomic<size_t> _iteration;
    std::atomic<bool> _toStop;
    std::string _lastError;
    std::atomic<fitness_funct_type> _bestFitnessFunction;
    std::future<void> _initFuture;
    std::future<void> _calcs;

    struct TRndData
    {
        cl_ulong s0;
        cl_ulong s1;
        cl_uint mat1;
        cl_uint mat2;
        cl_ulong tmat;
    };

    std::vector<TFitnessFunctOrder> _fitnessFunctOrder;

    fitness_funct_type _rejectFitnessFunctionLevel;
    fitness_funct_type _saveFitnessFunctionLevel;
    fitness_funct_type _stopFitnessFunctionLevel;
    bool _allRandomNets;

    std::vector<CNetStructureImpl::SNeuron> _neurons;
    cl::Buffer _neuronsCl;

    std::vector<TRndData> _randomBufferState;
    cl::Buffer _randomBufferStateCl;

    TDatas _netConfigs;
    cl::Buffer _netConfigsCl;

    TDatas _neuronsStates;
    cl::Buffer _neuronsStatesCl;

    TDatas _inputs;
    cl::Buffer _inputsCl;

    TOffsets _inputsOffs;
    cl::Buffer _inputsOffsCl;

    TDatas _outputs;
    cl::Buffer _outputsCl;

    TDatas _fitnessFunction;
    cl::Buffer _fitnessFunctionCl;

    TDatas _testDataInfos;
    cl::Buffer _testDataInfosCl;

    std::vector<char> _netsToRandomize;
    cl::Buffer _netsToRandomizeCl;

    TOffsets _crossingoverParent1;
    cl::Buffer _crossingoverParent1Cl;
    TOffsets _crossingoverParent2;
    cl::Buffer _crossingoverParent2Cl;

    TOffsets _mutationsAmounts;
    cl::Buffer _mutationsAmountsCl;

    cl::Platform _platform;
    cl::Device _device;
    cl::Context _context;
    std::unique_ptr<cl::Program> _clProg;
    cl::DeviceCommandQueue _deviceQueue;

    std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, TOffset, TOffset, cl::Buffer, TOffset>> _phase1Call_Randomizing;
    std::unique_ptr<cl::KernelFunctor<cl::Buffer, TOffset, TOffset, cl::Buffer, TOffset>> _phase1Call_RandomizeAll;
    std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, TOffset, TOffset, TOffset, cl::Buffer>> _phase2Call_Crossingover;
    std::unique_ptr<cl::KernelFunctor<cl::Buffer, TOffset, TOffset, TOffset>> _phase3Call_PrepareMutations;
    std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, TOffset, TOffset, TOffset, TOffset, cl::Buffer>> _phase4Call_Mutations;
    std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, TOffset, TOffset, TOffset, TOffset, TOffset, TOffset, TOffset>> _phase5Call_NetsCalculations;
    std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, TOffset, TOffset, TOffset, TOffset>> _phase6Call_CalcFitnessFunct;
    std::string _fitnessFunctionCalcName;

    std::size_t _phase1RandomizingThreads;
    std::size_t _phase1RandomizeAllThreads;
    std::size_t _phase2Threads;
    std::size_t _phase3Threads;
    std::size_t _phase4Threads;
    std::size_t _phase5Threads;
    std::size_t _phase6Threads;

    size_t _inputsPerNet;
    size_t _outputsPerNet;
    size_t _initialNets;
    size_t _configsAmount;
    size_t _initialRandoms;
    size_t _initialWorst;
    size_t _initialBest;
    size_t _crossingoverBrothers;
    std::vector<float> _mutations;
    size_t _initialStageAmount;
    size_t _crossingoverStageAmount;
    size_t _mutationsStageAmount;
    size_t _generalNetsAmount;
    size_t _randomRange;
    size_t _maxMutationsAmount;

    size_t _netConfigsBytes;
    size_t _allNetsConfigsBytes;

    std::mutex _configsToBeExportedMutex;
    std::vector<TDatas> _configsToBeExported;

    void Init(bool startCalculationsAutomatically);
    void CheckData();
    size_t GlobalMemoryBytesRequirement();
    void InitPlatform();
    void InitRandom();
    void InitNeuralNetworkStructure();
    void CalculateAmounts();
    void InitCalculations();
    void StartCalculations();
    bool CalcIteration();
    void InitState();

    void Phase1_RandomizeAll();
    void Phase1_Randomizing();
    void Phase2_Crossingover();
    void Phase3_PrepareMutations();
    void Phase4_Mutations();
    void Phase5_NetsCalculations();
    void Phase6_CalcFitnessFunct();
    void Phase7_SortNets();
    void Phase8_PrepareNets();
    void ExportGoodNetConfigs();

    template<typename T>
    void InitBuffer(T& vector, cl::Buffer& buffer, size_t size, bool readOnly, bool useHostPtr);

    template<typename T>
    void InitBuffer(T& vector, cl::Buffer& buffer, bool readOnly, bool useHostPtr);
};

}   // namespace artificial_neural_network

#endif // _AI_GENETIC_ALGORITHM_IMPL_HPP
