
#include "kernelCppHidings.hpp"

namespace artificial_neural_network::openCLemulator
{

#define tinymt64_seed(state, seed)
#define tinymt64_state int
#define tinymt64_uint64(value) 0
#define tinymt64wp_t int
#define TINYMT64_DOUBLE_MULTI 0
#define TestCheck() 0

#define TOffset ulong
#define TData double
#define TFitnessFunct double

/* KERNEL CODE BEGINNING */

typedef union
{
    TData _d;
    TOffset _u;
} UData;

typedef struct
{
    TOffset _inputsAmount;
    TOffset _firstInputOff;
    TOffset _firstConfigOff;
    TOffset _stateOff;
} SNeuron;

TOffset Index1of2(const TOffset index, const TOffset size1) { return index % size1; }
TOffset Index2of2(const TOffset index, const TOffset size1) { return index / size1; }

TOffset Index1of3(const TOffset index, const TOffset size1, const TOffset size2) { return index % size1; }
TOffset Index2of3(const TOffset index, const TOffset size1, const TOffset size2) { return (index / size1) % size2; }
TOffset Index3of3(const TOffset index, const TOffset size1, const TOffset size2) { return index / size1 / size2; }

TOffset Index1of4(const TOffset index, const TOffset size1, const TOffset size2, const TOffset size3) { return index % size1; }
TOffset Index2of4(const TOffset index, const TOffset size1, const TOffset size2, const TOffset size3) { return (index / size1) % size2; }
TOffset Index3of4(const TOffset index, const TOffset size1, const TOffset size2, const TOffset size3) { return (index / size1 / size2) % size3; }
TOffset Index4of4(const TOffset index, const TOffset size1, const TOffset size2, const TOffset size3) { return index / size1 / size2 / size3; }

size_t RandomOffset(void* randState, TOffset off)
{
    return tinymt64_uint64((tinymt64wp_t*)randState + off);
}

double RandomData(void* randState, TOffset off, TOffset randRange)
{
    double res = RandomOffset(randState, off) * TINYMT64_DOUBLE_MULTI;
    res *= 2 * randRange;
    res -= randRange;
    return res;
}

__kernel void Init_Random(__global void* randState, __global const ulong* randSeed)
{
    tinymt64_seed((tinymt64_state*)randState + get_global_id(0), randSeed[get_global_id(0)]);
}

#if TEST_GENETIC_ALGORITHM == 1
__kernel void Test_Random(__global void* randState, __global ulong* resInt, __global double* resDouble)
{
    const ulong off = get_global_id(0);
    resInt[off] = RandomOffset(randState, off);
    resDouble[off] = RandomData(randState, off, 10);
}
#endif // TEST_GENETIC_ALGORITHM

__kernel void Phase1_Randomizing
(
    __global const char* toRandomize, __global TData* netStructure,
    const TOffset netConfigItemsAmount, const TOffset netsAmount, __global void* randState, const TOffset randRange
)
{
    const TOffset netConfigItem = Index1of2(get_global_id(0), netConfigItemsAmount);
    const TOffset netId         = Index2of2(get_global_id(0), netConfigItemsAmount);

    if (!toRandomize[netId] || netConfigItem >= netConfigItemsAmount || netId >= netsAmount)
        return;

    const TOffset dataOff = netId * netConfigItemsAmount + netConfigItem;

    {
#if TEST_GENETIC_ALGORITHM == 1
        netStructure[get_global_id(0)] = netId + netConfigItem / 1000.0;
        return;
#elif TEST_GENETIC_ALGORITHM == 2
        netStructure[dataOff] = 1000000000000 + netId + netConfigItem / 1000.0;
        return;
#endif // TEST_GENETIC_ALGORITHM
    }

    netStructure[dataOff] = RandomData(randState, get_global_id(0), randRange);
}

__kernel void Phase1_RandomizeAll
(
    __global TData* netStructure,
    const TOffset netConfigItemsAmount, const TOffset netsAmount, __global void* randState, const TOffset randRange
)
{
    const TOffset netConfigItem = Index1of2(get_global_id(0), netConfigItemsAmount);
    const TOffset netId         = Index2of2(get_global_id(0), netConfigItemsAmount);

    if (netConfigItem >= netConfigItemsAmount || netId >= netsAmount)
        return;

    const TOffset dataOff = netId * netConfigItemsAmount + netConfigItem;
    netStructure[dataOff] = RandomData(randState, get_global_id(0), randRange);
}

__kernel void Phase2_Crossingover
(
    __global TData* netStructure, __global const TOffset* parent1Val, __global const TOffset* parent2Val,
    const TOffset netConfigItemsAmount, const TOffset parentsAmount, const TOffset childrenAmount, __global void* randState
)
{
    const TOffset netConfigItem = Index1of3(get_global_id(0), netConfigItemsAmount, childrenAmount);
    const TOffset children      = Index2of3(get_global_id(0), netConfigItemsAmount, childrenAmount);
    const TOffset parents       = Index3of3(get_global_id(0), netConfigItemsAmount, childrenAmount);
    const TOffset parent1 = parent1Val[parents];
    const TOffset parent2 = parent2Val[parents];

    const TOffset parentsCombinations = parentsAmount * parentsAmount / 2 - parentsAmount / 2;
    if (netConfigItem >= netConfigItemsAmount || children >= childrenAmount || parents >= parentsCombinations)
        return;

    const TOffset parent1Off = netConfigItemsAmount * parent1 + netConfigItem;
    const TOffset parent2Off = netConfigItemsAmount * parent2 + netConfigItem;
    const TOffset childrenOff = netConfigItemsAmount * (parentsAmount + parents * childrenAmount + children) + netConfigItem;

    const TOffset off = childrenOff;

    {
#if TEST_GENETIC_ALGORITHM == 1
        netStructure[get_global_id(0)] = parent1*1000000 + parent2*1000 + children + netConfigItem / 1000.0;
        return;
#elif TEST_GENETIC_ALGORITHM == 2
        TOffset parent1Value = (TOffset)(netStructure[parent1Off] * 1000) % 1000000;
        TOffset parent2Value = (TOffset)(netStructure[parent2Off] * 1000) % 1000000;
        TOffset netConfigItemValue = (parent1Value + parent2Value - netConfigItem) % 1000;
        parent1Value /= 1000;
        parent2Value /= 1000;
        const TData value = 2000000000000 + parent1Value * 1000000 + parent2Value * 1000 + children + netConfigItemValue / 1000.0;
        netStructure[off] = value;
        return;
#endif // TEST_GENETIC_ALGORITHM
    }

    UData parent1Value;   parent1Value._d = netStructure[parent1Off];
    UData parent2Value;   parent2Value._d = netStructure[parent2Off];

    TOffset sel = RandomOffset(randState, get_global_id(0));

    UData res;
    res._u = (parent1Value._u & sel) | (parent2Value._u & ~sel);

    netStructure[off] = res._d;
}

__kernel void Phase3_PrepareMutations
(
    __global TData* netStructure,
    const TOffset netConfigItemsAmount, const TOffset sourcesAmount, const TOffset mutationsAmount
)
{
    const TOffset netConfigItem = Index1of3(get_global_id(0), netConfigItemsAmount, sourcesAmount);
    const TOffset source        = Index2of3(get_global_id(0), netConfigItemsAmount, sourcesAmount);
    const TOffset mutation      = Index3of3(get_global_id(0), netConfigItemsAmount, sourcesAmount);
    const TOffset sourceOff = source * netConfigItemsAmount + netConfigItem;
    const TOffset destOff = (sourcesAmount + source * mutationsAmount + mutation) * netConfigItemsAmount + netConfigItem;

    if (netConfigItem >= netConfigItemsAmount || source >= sourcesAmount || mutation >= mutationsAmount)
        return;

    {
#if TEST_GENETIC_ALGORITHM == 1
        netStructure[get_global_id(0)] = 3000000000000000 + mutation * 100000000000000 + destOff * 100000000 + sourceOff * 10000 + source;
        return;
#elif TEST_GENETIC_ALGORITHM == 2
        netStructure[destOff] = 3000000000000000 + get_global_id(0);
        return;
#endif // TEST_GENETIC_ALGORITHM
    }

    netStructure[destOff] = netStructure[sourceOff];
}

__kernel void Phase4_Mutations
(
    __global TData* netStructure, __global TOffset* mutationsAmounts, const TOffset firstNetConfigOff,
    const TOffset netConfigItemsAmount, const TOffset netsAmount, const TOffset mutationsAmount, __global void* randState
)
{
    const TOffset net           = Index1of3(get_global_id(0), netsAmount, mutationsAmount);
    const TOffset mutation      = Index2of3(get_global_id(0), netsAmount, mutationsAmount);
    const TOffset mutationStep  = Index3of3(get_global_id(0), netsAmount, mutationsAmount);
    const TOffset netOff = firstNetConfigOff + net * netConfigItemsAmount;

#if TEST_GENETIC_ALGORITHM == 1
    netStructure[get_global_id(0)] = 0;
#endif // TEST_GENETIC_ALGORITHM

    if (net >= netsAmount || mutation >= mutationsAmount || mutationStep >= mutationsAmounts[mutation])
        return;

    {
#if TEST_GENETIC_ALGORITHM == 1
        netStructure[get_global_id(0)] = 4000000000000000 + netOff * 100000000 + mutationStep * 100000 + mutation * 10000 + net;
        return;
#elif TEST_GENETIC_ALGORITHM == 2
        netStructure[netOff + mutation] = 4000000000000000 + mutationsAmounts[mutation] * 1000000000 + net;
        return;
#endif // TEST_GENETIC_ALGORITHM
    }

    const TOffset byte = RandomOffset(randState, get_global_id(0)) % netConfigItemsAmount;
    const TOffset bit = RandomOffset(randState, get_global_id(0)) % (8 * sizeof(TData));

    const TOffset dataOff = netOff + byte;
    const size_t dataSelector = ((size_t)1) << bit;

    UData value;
    value._d = netStructure[dataOff];
    value._u = (value._u & ~dataSelector) | (~value._u & dataSelector);

    netStructure[dataOff] = value._d;
};

__global const TData* ProcessNeuron
(
    __global const SNeuron* neuron,
    __global const TOffset* inputsOffs,
    __global const TData* inputs,
    __global TData* neuronsStates,
    __global const TData* config,
    __global TData* outputs
   )
{
    const TOffset externalDirBit = ((TOffset)1) << (sizeof(TOffset) * 8 - 1);

    TOffset inputOffPtr = neuron->_firstInputOff;
    TData sum = 0;
    config += neuron->_firstConfigOff;
    for (TOffset i = 0; i < neuron->_inputsAmount; ++i, ++inputOffPtr)
    {
        const TOffset inputOff = inputsOffs[inputOffPtr];
        const bool isExternalInput = inputOff & externalDirBit;
        const TOffset unmaskedInputOff = inputOff & ~externalDirBit;

        const TData inputValue = isExternalInput ? inputs[unmaskedInputOff] : neuronsStates[unmaskedInputOff];
        const TData curInputValue = inputValue * *config++;
        sum += curInputValue;
    }

    typedef enum { IDENTITY = 0, SIGMOID, BINARY_STEP_PARAM, BINARY_STEP, IDENTITY_PARAM, TANH, RELU, RELU_PARAM, SOFTPLUS, ELU, ELU_PARAM, SELU, LRELU, SILU, GAUSSIAN, ACTIVATION_FUNCTIONS_AMOUNT } EActivationFunction;
#if TEST_GENETIC_ALGORITHM == 1
    const EActivationFunction actFunct = *config++ == 0 ? IDENTITY : SIGMOID;
#else // TEST_GENETIC_ALGORITHM == 1
    const EActivationFunction actFunct = (EActivationFunction)(abs((ulong)(*config++ * ACTIVATION_FUNCTIONS_AMOUNT)) % ACTIVATION_FUNCTIONS_AMOUNT);
#endif // TEST_GENETIC_ALGORITHM == 1

    const TData actFunctParam1 = *config++;
    const TData actFunctParam2 = *config++;

    const double e_plus_sum = exp(sum);
    const double e_minus_sum = exp(-sum);

    double res;
    switch (actFunct)
    {
    case IDENTITY:          res = sum; break;
    case SIGMOID:           res = 1 / (1 + e_minus_sum); break;
    case BINARY_STEP_PARAM: res = (sum < actFunctParam1 ? 0 : 1); break;
    case BINARY_STEP:       res = (sum < 0 ? 0 : 1); break;
    case IDENTITY_PARAM:    res = sum * actFunctParam1 + actFunctParam2; break;
    case TANH:              res = (e_plus_sum - e_minus_sum) / (e_plus_sum + e_minus_sum); break;
    case RELU:              res = (sum <= 0 ? 0 : sum); break;
    case RELU_PARAM:        res = (sum < actFunctParam2 ? actFunctParam1 * sum : sum); break;
    case SOFTPLUS:          res = log(1 + e_plus_sum); break;
    case ELU:               res = (sum < 0 ? actFunctParam1 * (e_plus_sum - 1) : sum); break;
    case ELU_PARAM:         res = (sum < actFunctParam2 ? actFunctParam1 * (e_plus_sum - 1) : sum); break;
    case SELU:              res = actFunctParam2 * (sum < 0 ? actFunctParam1 * (e_plus_sum - 1) : sum); break;
    case LRELU:             res = (sum < 0 ? 0.01 * sum : sum); break;
    case SILU:              res = sum / (1 + e_minus_sum); break;
    case GAUSSIAN:          res = exp(-(sum * sum)); break;
    default:                res = -1.7976931348623157e+308;
    }

    const TOffset resOff = neuron->_stateOff;
    const bool isExternalOutput = resOff & externalDirBit;
    const TOffset unmaskedResOff = resOff & ~externalDirBit;

    if (isExternalOutput)
        outputs[unmaskedResOff] = res;
    else
        neuronsStates[unmaskedResOff] = res;

    return config;
}

__kernel void Phase5_NetsCalculations
(
    __global const TData* netStructure, __global const TOffset* inputsOffs, __global const TData* inputs, __global TData* outputs, __global const SNeuron* neurons, __global TData* neuronsStates,
    const TOffset netConfigItemsAmount, const TOffset netsAmount, const TOffset testSetsAmount, const TOffset inputsPerNetwork, const TOffset outputsPerNetwork, const TOffset neuronsPerNetwork, const TOffset neuronsStatesPerNetwork
)
{
    const TOffset dataSet = Index1of2(get_global_id(0), netsAmount);
    const TOffset net = Index2of2(get_global_id(0), netsAmount);

    if (net >= netsAmount || dataSet >= testSetsAmount)
        return;

    const TOffset inputsOff = dataSet * inputsPerNetwork;
    const TOffset netStructureOff = net * netConfigItemsAmount;
    const TOffset outputsOff = ( net * testSetsAmount + dataSet ) * outputsPerNetwork;
    const TOffset neuronsStatesOff = ( net * testSetsAmount + dataSet ) * neuronsStatesPerNetwork;

    const TData* neuronConfigData = netStructure + netStructureOff;
    for (TOffset i = 0; i < neuronsPerNetwork; ++i)
        neuronConfigData = ProcessNeuron(neurons + i, inputsOffs, inputs + inputsOff, neuronsStates + neuronsStatesOff, netStructure + netStructureOff, outputs + outputsOff);
}

// TData TestCheck(__global const TData* outputs, __global const TData* testData, const TOffset datasetsAmount);

__kernel void Phase6_CalcFitnessFunct(
    __global const TData* outputs, __global const TData* datasetsValue, __global TData* fitnessFunct,
    const TOffset outputsPerNet, const TOffset testDataPerDataset, const TOffset datasetsAmount, const TOffset netsAmount
)
{
    const TOffset net = get_global_id(0);
    if (net >= netsAmount)
        return;

    __global const TData* curOutputs = outputs + outputsPerNet * datasetsAmount * net;

    fitnessFunct[net] = TestCheck(curOutputs, datasetsValue, datasetsAmount);
}

/* KERNEL CODE ENDING */

}   // namespace artificial_neural_network::openCLemulator
