
#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <unordered_set>
#include "geneticAlgorithmImpl.hpp"

using namespace artificial_neural_network;
using namespace std::string_literals;

void CGeneticAlgorithmImpl::StartCalculations()
{
    while (true)
    {
        const bool toContinue = CalcIteration();
        if (!toContinue)
            break;

        if constexpr (_testGeneticAlgorithm >= 1)
            break;
    }

    _status = status::got_result;
}

bool CGeneticAlgorithmImpl::CalcIteration()
{
    _iteration = 0;

    bool firstIteration = true;
    try
    {
        do
        {
            if (firstIteration)
                firstIteration = false;
            else
                ++_iteration;

            _status = status::calc_phase1;
            if (_allRandomNets)
            {
                _status = status::calc_phase1;  Phase1_RandomizeAll();
            }
            else
            {
                _status = status::calc_phase1;  Phase1_Randomizing();
                _status = status::calc_phase2;  Phase2_Crossingover();
                _status = status::calc_phase3;  Phase3_PrepareMutations();
                _status = status::calc_phase4;  Phase4_Mutations();
            }

            _status = status::calc_phase5;  Phase5_NetsCalculations();
            _status = status::calc_phase6;  Phase6_CalcFitnessFunct();
            _status = status::calc_phase7;  Phase7_SortNets();
            _status = status::calc_phase8;  Phase8_PrepareNets();
        }
        while (_bestFitnessFunction < _stopFitnessFunctionLevel && !_toStop);

    }
    catch (std::runtime_error& err)
    {
        _status = status::calc_error;
        _lastError = err.what();
    }

    return true;
}

void CGeneticAlgorithmImpl::Phase1_RandomizeAll()
{
    cl_int error;

    _phase1Call_RandomizeAll->operator()(cl::EnqueueArgs(cl::NDRange(_phase1RandomizeAllThreads)),
        _netConfigsCl,
        _configsAmount, _generalNetsAmount, _randomBufferStateCl, _randomRange,
        error);
    if (error != CL_SUCCESS)
        throw std::runtime_error("Phase 1 error : error code "s + std::to_string(error));

    if constexpr (_testGeneticAlgorithm == 1)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());
    }
    else if constexpr (_testGeneticAlgorithm == 3)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());
    }
}

void CGeneticAlgorithmImpl::Phase1_Randomizing()
{
    cl_int error;

    _phase1Call_Randomizing->operator()( cl::EnqueueArgs(cl::NDRange(_phase1RandomizingThreads)),
        _netsToRandomizeCl, _netConfigsCl,
        _configsAmount, _initialNets, _randomBufferStateCl, _randomRange,
        error);
    if (error != CL_SUCCESS)
        throw std::runtime_error("Phase 1 error : error code "s + std::to_string(error));

    if constexpr (_testGeneticAlgorithm == 1)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;
        for (size_t netId = 0; netId < _initialNets; ++netId)
            for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
                assert(_netConfigs[off++] == netId + netConfigItem / 1000.0);
    }
    else if constexpr (_testGeneticAlgorithm == 2)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;
        for (size_t netId = 0; netId < _initialNets; ++netId)
            for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
            {
                if (_netsToRandomize[netId])
                    assert(_netConfigs[off] == 1000000000000 + netId + netConfigItem/1000.0);
                else
                    assert(_netConfigs[off] == 0);
                ++off;
            }

        for (; off < _configsAmount * _generalNetsAmount; ++off)
            assert(_netConfigs[off] == 0);
    }
    else if constexpr (_testGeneticAlgorithm == 3)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());
    }
 }

void CGeneticAlgorithmImpl::Phase2_Crossingover()
{
    cl_int error;

    _phase2Call_Crossingover->operator()( cl::EnqueueArgs(cl::NDRange(_phase2Threads)),
        _netConfigsCl, _crossingoverParent1Cl, _crossingoverParent2Cl,
        _configsAmount, _initialNets, _crossingoverBrothers, _randomBufferStateCl,
        error);
    if (error != CL_SUCCESS)
        throw std::runtime_error("Phase 2 error : error code "s + std::to_string(error));

    if constexpr (_testGeneticAlgorithm == 1)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;
        for (size_t parentId = 0; parentId < _crossingoverStageAmount / _crossingoverBrothers; ++parentId)
            for (size_t childrenId = 0; childrenId < _crossingoverBrothers; ++childrenId)
                for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
                {
                    const TData actual = _netConfigs[off];
                    const TOffset parent1 = _crossingoverParent1[parentId];
                    const TOffset parent2 = _crossingoverParent2[parentId];
                    const TData expected = parent1 * 1'000'000 + parent2 * 1'000 + childrenId + netConfigItem / 1000.0;
                    assert(expected == actual);
                    off++;
                }
    }
    else if constexpr (_testGeneticAlgorithm == 2)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;

        for (size_t netId = 0; netId < _initialNets; ++netId)
            for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
            {
                if (_netsToRandomize[netId])
                    assert(_netConfigs[off] == 1000000000000 + netId + netConfigItem/1000.0);
                else
                    assert(_netConfigs[off] == 0);
                ++off;
            }

        for (size_t parentId = 0; parentId < _crossingoverStageAmount / _crossingoverBrothers; ++parentId)
            for (size_t children = 0; children < _crossingoverBrothers; ++children)
                for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
                {
                    const TData actual = _netConfigs[off];

                    const TOffset parent1 = _crossingoverParent1[parentId];
                    const TOffset parent2 = _crossingoverParent2[parentId];
                    const TOffset netConfigItemAct = static_cast<TOffset>(actual * 1'000) % 1'000;
                    const TOffset childrenAct = static_cast<TOffset>(actual) % 1'000;
                    const TOffset parent2Act = static_cast<TOffset>(actual) / 1'000 % 1'000;
                    const TOffset parent1Act = static_cast<TOffset>(actual) / 1'000'000 % 1'000;
                    const TOffset dataType = static_cast<TOffset>(actual) / 1'000'000'000'000;

                    assert(netConfigItem == netConfigItemAct);
                    assert(parent1 == parent1Act);
                    assert(parent2 == parent2Act);
                    assert(children == childrenAct);
                    assert(dataType == 2);

                    ++off;
                }

        for (; off < _configsAmount * _generalNetsAmount; ++off)
            assert(_netConfigs[off] == 0);
    }
    else if constexpr (_testGeneticAlgorithm == 3)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());
    }
}

void CGeneticAlgorithmImpl::Phase3_PrepareMutations()
{
    cl_int error;
    
    const size_t sourcesAmount = _initialNets + _crossingoverStageAmount;
    const size_t mutationsAmount = _mutations.size();

    _phase3Call_PrepareMutations->operator()( cl::EnqueueArgs(cl::NDRange(_phase3Threads)),
        _netConfigsCl,
        _configsAmount, sourcesAmount, mutationsAmount,
        error);
    if (error != CL_SUCCESS)
        throw std::runtime_error("Phase 3 error : error code "s + std::to_string(error));

    if constexpr (_testGeneticAlgorithm == 1)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;
        for (size_t mutationId = 0; mutationId < mutationsAmount; ++mutationId)
            for (size_t sourceId = 0; sourceId < sourcesAmount; ++sourceId)
                for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
                {
                    const TData actual = _netConfigs[off];

                    const TOffset mutationActual = static_cast<TOffset>(actual / 100000000000000) % 10;
                    const TOffset sourceActual = static_cast<TOffset>(actual) % 1000;

                    const TOffset sourceOffActual = static_cast<TOffset>(actual / 10000) % 10000;
                    const TOffset destOffActual = static_cast<TOffset>(actual / 100000000) % 1000000;

                    const TOffset sourceOffExpected = sourceId * _configsAmount + netConfigItem;
                    const TOffset destOffExpected = (sourcesAmount + sourceId * mutationsAmount + mutationId) * _configsAmount + netConfigItem;

                    const TOffset dataType = static_cast<TOffset>(actual) / 1000000000000000;

                    assert(mutationId == mutationActual);
                    assert(sourceId == sourceActual);
                    assert(sourceOffActual == sourceOffExpected);
                    assert(destOffActual == destOffExpected);
                    assert(dataType == 3);

                    off++;
                }
    }
    else if constexpr (_testGeneticAlgorithm == 2)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;

        for (size_t netId = 0; netId < _initialNets; ++netId)
            for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
            {
                if (_netsToRandomize[netId])
                    assert(_netConfigs[off] == 1000000000000 + netId + netConfigItem/1000.0);
                else
                    assert(_netConfigs[off] == 0);
                ++off;
            }

        for (size_t parentId = 0; parentId < _crossingoverStageAmount / _crossingoverBrothers; ++parentId)
            for (size_t children = 0; children < _crossingoverBrothers; ++children)
                for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
                {
                    const TData actual = _netConfigs[off];

                    const TOffset parent1 = _crossingoverParent1[parentId];
                    const TOffset parent2 = _crossingoverParent2[parentId];
                    const TOffset netConfigItemAct = static_cast<TOffset>(actual * 1'000) % 1'000;
                    const TOffset childrenAct = static_cast<TOffset>(actual) % 1'000;
                    const TOffset parent2Act = static_cast<TOffset>(actual) / 1'000 % 1'000;
                    const TOffset parent1Act = static_cast<TOffset>(actual) / 1'000'000 % 1'000;
                    const TOffset dataType = static_cast<TOffset>(actual) / 1'000'000'000'000;

                    assert(netConfigItem == netConfigItemAct);
                    assert(parent1 == parent1Act);
                    assert(parent2 == parent2Act);
                    assert(children == childrenAct);
                    assert(dataType == 2);

                    ++off;
                }

        std::unordered_map<size_t, size_t> values;
        for (; off < _configsAmount * _generalNetsAmount; ++off)
            ++values[static_cast<TOffset>(_netConfigs[off]) % 1000000000000000];

        for (auto it : values)
            assert(it.second == 1);
    }
    else if constexpr (_testGeneticAlgorithm == 3)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());
    }
}

void CGeneticAlgorithmImpl::Phase4_Mutations()
{
    cl_int error;
    
    const size_t sourcesAmount = _initialNets + _crossingoverStageAmount;
    const size_t mutationsAmount = _mutations.size();
    const size_t firstNetConfigOff = (_initialNets + _crossingoverStageAmount) * _configsAmount;

    _phase4Call_Mutations->operator()( cl::EnqueueArgs(cl::NDRange(_phase4Threads)),
        _netConfigsCl, _mutationsAmountsCl, firstNetConfigOff,
        _configsAmount, sourcesAmount, _mutations.size(), _randomBufferStateCl,
        error);
    if (error != CL_SUCCESS)
        throw std::runtime_error("Phase 4 error : error code "s + std::to_string(error));

    if constexpr (_testGeneticAlgorithm == 1)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;
        for (size_t mutationStepId = 0; mutationStepId < _maxMutationsAmount; ++mutationStepId)
            for (size_t mutationId = 0; mutationId < _mutations.size(); ++mutationId)
                for (size_t net = 0; net < sourcesAmount; ++net)
                {
                    const TData actual = _netConfigs[off];
                    ++off;

                    if (mutationStepId >= _mutationsAmounts[mutationId])
                    {
                        assert(actual == 0);
                        continue;
                    }

                    const TOffset netActual          = static_cast<TOffset>(actual) % 10000;
                    const TOffset mutationActual     = (static_cast<TOffset>(actual) /            10000) % 10;
                    const TOffset mutationStepActual = (static_cast<TOffset>(actual) /           100000) % 1000;
                    const TOffset netOffActual       = (static_cast<TOffset>(actual) /        100000000) % 100000;
                    const TOffset dataType           =  static_cast<TOffset>(actual) / 1000000000000000;

                    assert(netActual == net);
                    assert(mutationActual == mutationId);
                    assert(mutationStepActual == mutationStepId);
                    assert(netOffActual == netOffActual);
                    assert(dataType == 4);
                }
    }
    else if constexpr (_testGeneticAlgorithm == 2)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());

        size_t off = 0;

        for (size_t netId = 0; netId < _initialNets; ++netId)
            for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
            {
                if (_netsToRandomize[netId])
                    assert(_netConfigs[off] == 1000000000000 + netId + netConfigItem / 1000.0);
                else
                    assert(_netConfigs[off] == 0);
                ++off;
            }

        for (size_t parentId = 0; parentId < _crossingoverStageAmount / _crossingoverBrothers; ++parentId)
            for (size_t children = 0; children < _crossingoverBrothers; ++children)
                for (size_t netConfigItem = 0; netConfigItem < _configsAmount; ++netConfigItem)
                {
                    const TData actual = _netConfigs[off];

                    const TOffset parent1 = _crossingoverParent1[parentId];
                    const TOffset parent2 = _crossingoverParent2[parentId];
                    const TOffset netConfigItemAct =  static_cast<TOffset>(actual  * 1'000) % 1'000;
                    const TOffset childrenAct      =  static_cast<TOffset>(actual) % 1'000;
                    const TOffset parent2Act       = (static_cast<TOffset>(actual) /             1'000) % 1'000;
                    const TOffset parent1Act       = (static_cast<TOffset>(actual) /         1'000'000) % 1'000;
                    const TOffset dataType         =  static_cast<TOffset>(actual) / 1'000'000'000'000;

                    assert(netConfigItem == netConfigItemAct);
                    assert(parent1 == parent1Act);
                    assert(parent2 == parent2Act);
                    assert(children == childrenAct);
                    assert(dataType == 2);

                    ++off;
                }

        for (size_t mutation = 0; mutation < mutationsAmount; ++mutation)
            for (size_t net = 0; net < _mutationsStageAmount; ++net)
            {
                off = firstNetConfigOff + net * _configsAmount + mutation;
                const TData actual = _netConfigs[off];

                const TOffset dataType = static_cast<TOffset>(actual / 1000000000000000);
                const TOffset netActual = static_cast<TOffset>(actual) % 1000000000;
                const TOffset mutationsAmountActual = (static_cast<TOffset>(actual) / 1000000000) % 1000000;

                assert(dataType == 4);
                assert(mutationsAmountActual == _mutationsAmounts[mutation]);
                assert(netActual == net);
            }
    }
    else if constexpr (_testGeneticAlgorithm == 3)
    {
        cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());
    }
}

void CGeneticAlgorithmImpl::Phase5_NetsCalculations()
{
    cl_int error;

    if constexpr (_testGeneticAlgorithm == 1)
    {
        _generalNetsAmount = 4;
        _netConfigs = { 1,0,0,0,0.5,1,1,1,2.5,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,10,1,1,1,1,5,3,0,0,0,-2,1.45,8,0,0,0,0,1,0,0,0,1,1,0,0,0,8.5,0,0,0,-3,1,1,1,1.34,1,1,1,8,-0.3,0,0,0,0,1.45,2.2,1,0,0,0,-0.3,-1,0,0,0,0.3,0.8,0,0,0,12,0,0,0,-8,1,1,1,5.3,0,0,0,-12,15,2,0,0,0,8.3,-15.7,-7,1,1,1,12.7,8.9,0,0,0,-14,-2,1,1,1 };
        cl::copy(_netConfigs.begin(), _netConfigs.end(), _netConfigsCl);
    }

    _phase5Call_NetsCalculations->operator()( cl::EnqueueArgs(cl::NDRange(_phase5Threads)),
        _netConfigsCl, _inputsOffsCl, _inputsCl, _outputsCl, _neuronsCl, _neuronsStatesCl,
        _configsAmount, _generalNetsAmount, _testDatas.size(), _inputsPerNet, _outputsPerNet,  _netStructure._neurons.size(), _netStructure._statesSize,
        error);
    if (error != CL_SUCCESS)
        throw std::runtime_error("Phase 5 error : error code "s + std::to_string(error));

    if constexpr (_testGeneticAlgorithm == 1)
    {
        auto cmp = [](TData a, TData b, TData err = 0.1) { return std::round(std::abs(a - b)) <= err; };

        cl::copy(_outputsCl, _outputs.begin(), _outputs.end());

        assert(cmp(_outputs[0], 0.9));
        assert(cmp(_outputs[1], 5.2));

        assert(cmp(_outputs[2], 1));
        assert(cmp(_outputs[3], 13.5));

        assert(cmp(_outputs[4], 1));
        assert(cmp(_outputs[5], 18));

        assert(cmp(_outputs[6], 9.1));
        assert(cmp(_outputs[7], 15.7));

        assert(cmp(_outputs[8], 9.3));
        assert(cmp(_outputs[9], 16.7));

        assert(cmp(_outputs[10], 8));
        assert(cmp(_outputs[11], 11));

        assert(cmp(_outputs[12], -34));
        assert(cmp(_outputs[13], 31.1));

        assert(cmp(_outputs[14], -165));
        assert(cmp(_outputs[15], 152.1));

        assert(cmp(_outputs[16], -265));
        assert(cmp(_outputs[17], 244.5));

        assert(cmp(_outputs[18], -1550.6));
        assert(cmp(_outputs[19], 1));

        assert(cmp(_outputs[20], -8192.8));
        assert(cmp(_outputs[21], 1));

        assert(cmp(_outputs[22], -13623.3));
        assert(cmp(_outputs[23], 1));
    }
    else if constexpr (_testGeneticAlgorithm == 3)
    {
        cl::copy(_neuronsStatesCl, _neuronsStates.begin(), _neuronsStates.end());
        cl::copy(_outputsCl, _outputs.begin(), _outputs.end());
    }

}

void CGeneticAlgorithmImpl::Phase6_CalcFitnessFunct()
{
    cl_int error;

    _phase6Call_CalcFitnessFunct->operator()( cl::EnqueueArgs(cl::NDRange(_phase6Threads)),
        _outputsCl, _testDataInfosCl, _fitnessFunctionCl,
        _outputsPerNet, _testDatas.at(0).interpret_data.size(), _testDatas.size(), _generalNetsAmount,
        error);
    if (error != CL_SUCCESS)
        throw std::runtime_error("Phase 6 error : error code "s + std::to_string(error));

    if constexpr (_testGeneticAlgorithm == 1)
    {
        cl::copy(_fitnessFunctionCl, _fitnessFunction.begin(), _fitnessFunction.end());

        auto cmp = [](TData a, TData b, TData err = 0.1) { return std::round(std::abs(a - b)) <= err; };

        assert(cmp(_fitnessFunction[0], 33.6));
        assert(cmp(_fitnessFunction[1], 75.2));
        assert(cmp(_fitnessFunction[2], -49));
        assert(cmp(_fitnessFunction[3], -24755.3));
    }
    else if constexpr (_testGeneticAlgorithm == 3)
    {
        cl::copy(_fitnessFunctionCl, _fitnessFunction.begin(), _fitnessFunction.end());
    }
}

void CGeneticAlgorithmImpl::Phase7_SortNets()
{
    cl::copy(_fitnessFunctionCl, _fitnessFunction.begin(), _fitnessFunction.end());

    for (size_t i = 0; i < _fitnessFunction.size(); ++i)
        _fitnessFunctOrder[i] = {i, _fitnessFunction[i]};

    std::sort(std::execution::par, _fitnessFunctOrder.begin(), _fitnessFunctOrder.end(), [this](const TFitnessFunctOrder& a, const TFitnessFunctOrder& b)
    {
        return a.second > b.second;
    });
}

void CGeneticAlgorithmImpl::Phase8_PrepareNets()
{
    size_t firstBadNetPos;
    for (firstBadNetPos = 0; firstBadNetPos < _fitnessFunctOrder.size(); ++firstBadNetPos)
        if (_fitnessFunctOrder[firstBadNetPos].second <= _rejectFitnessFunctionLevel)
            break;

    _allRandomNets = firstBadNetPos == 0;
    _bestFitnessFunction = _fitnessFunctOrder[0].second;
    if (_bestFitnessFunction >= _stopFitnessFunctionLevel)
        return;

    if (_allRandomNets)
        return;

    std::fill(_netsToRandomize.begin(), _netsToRandomize.end(), true);

    cl::copy(_netConfigsCl, _netConfigs.begin(), _netConfigs.end());
    TDatas netConfigsTmp;
    netConfigsTmp.resize(_configsAmount * _initialNets);

    size_t netsToCopy = std::min(firstBadNetPos, _initialNets - _initialWorst - _initialRandoms);
    auto dstTmp = netConfigsTmp.begin();

    size_t i = 0;
    for (; i < netsToCopy; ++i)
    {
        _netsToRandomize[i] = false;
        const size_t netSrc = _fitnessFunctOrder[i].first;
        auto src = _netConfigs.begin() + netSrc * _configsAmount;
        for (size_t j = 0; j < _configsAmount; ++j)
            *dstTmp++ = *src++;
    }

    if (netsToCopy < (firstBadNetPos - _initialWorst))
    {
        for (i = 0; i < _initialWorst; ++i)
        {
            _netsToRandomize[netsToCopy + i] = false;
            const size_t netSrc = _fitnessFunctOrder[firstBadNetPos - i].first;
            auto src = _netConfigs.begin() + netSrc * _configsAmount;
            for (size_t j = 0; j < _configsAmount; ++j)
                *dstTmp++ = *src++;
        }
    }

    if constexpr (_testGeneticAlgorithm == 3)
    {
        auto src = netConfigsTmp.begin();
        auto dst = _netConfigs.begin();

        for (size_t net = 0; net < i; ++net)
            for (size_t j = 0; j < _configsAmount; ++j)
                *dst++ = *src++;
    }
    cl::copy(netConfigsTmp.begin(), netConfigsTmp.end(), _netConfigsCl);

    cl::copy(_netsToRandomize.begin(), _netsToRandomize.end(), _netsToRandomizeCl);
}
