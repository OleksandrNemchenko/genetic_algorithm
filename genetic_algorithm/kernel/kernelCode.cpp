
#include <string>
#include "../geneticAlgorithmImpl.hpp"

/* static */ const std::string artificial_neural_network::CGeneticAlgorithmImpl::_definesProg = R"(


#define TOffset ulong
#define TData double
#define TFitnessFunct double


)";

/* static */ const std::string artificial_neural_network::CGeneticAlgorithmImpl::_geneticAlgProg = R"(


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


)";

/* static */ const std::string artificial_neural_network::CGeneticAlgorithmImpl::_randomProg = R"(


#define HAVE_DOUBLE

/**
@file

Implements RandomCL interface to tinymt64 RNG.

Tiny mersenne twister, http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/index.html.
*/

#define TINYMT64_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TINYMT64_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define KERNEL_PROGRAM

/**
 * return unique id in a device.
 * This function may not work correctly in multiple devices.
 * @return unique id in a device
 */
inline static size_t
tinymt_get_sequential_id()
{
    return get_global_id(2) - get_global_offset(2)
        + get_global_size(2) * (get_global_id(1) - get_global_offset(1))
        + get_global_size(1) * get_global_size(2)
        * (get_global_id(0) - get_global_offset(0));
}

/**
 * return number of unique ids in a device.
 * This function may not work correctly in multiple devices.
 * @return number of unique ids in a device
 */
inline static size_t
tinymt_get_sequential_size()
{
    return get_global_size(0) * get_global_size(1) * get_global_size(2);
}

#if defined(KERNEL_PROGRAM)
#if !defined(cl_uint)
#define cl_uint uint
#endif
#if !defined(cl_ulong)
#define cl_ulong ulong
#endif
#if !defined(UINT64_X)
#define UINT64_C(x) (x ## UL)
#endif
#endif

/**
 * TinyMT32 structure with parameters
 */
typedef struct TINYMT64WP_T {
    cl_ulong s0;
    cl_ulong s1;
    cl_uint mat1;
    cl_uint mat2;
    cl_ulong tmat;
} tinymt64wp_t;

/**
 * TinyMT32 structure for jump without parameters
 */
typedef struct TINYMT64J_T {
    cl_ulong s0;
    cl_ulong s1;
} tinymt64j_t;

#define TINYMT64J_MAT1 0xfa051f40U
#define TINYMT64J_MAT2 0xffd0fff4U;
#define TINYMT64J_TMAT UINT64_C(0x58d02ffeffbfffbc)

#define TINYMT64_SHIFT0 12
#define TINYMT64_SHIFT1 11
#define TINYMT64_MIN_LOOP 8

__constant ulong tinymt64_mask = 0x7fffffffffffffffUL;
__constant ulong tinymt64_double_mask = 0x3ff0000000000000UL;

#if defined(HAVE_DOUBLE)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 */
inline static void
tinymt64_next_state(tinymt64wp_t * tiny)
{
    ulong x;

    tiny->s0 &= tinymt64_mask;
    x = tiny->s0 ^ tiny->s1;
    x ^= x << TINYMT64_SHIFT0;
    x ^= x >> 32;
    x ^= x << 32;
    x ^= x << TINYMT64_SHIFT1;
    tiny->s0 = tiny->s1;
    tiny->s1 = x;
    if (x & 1) {
        tiny->s0 ^= tiny->mat1;
        tiny->s1 ^= (ulong)tiny->mat2 << 32;
    }
}

/**
 * tempering output function
 *@param tiny internal state of tinymt with parameter
 *@return tempered output
 */
inline static ulong
tinymt64_temper(tinymt64wp_t * tiny)
{
    ulong x;
    x = tiny->s0 + tiny->s1;
    x ^= tiny->s0 >> 8;
    if (x & 1) {
        x ^= tiny->tmat;
    }
    return x;
}
/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 *@return 32-bit random integer
 */
inline static ulong
tinymt64_uint64(tinymt64wp_t * tiny)
{
    tinymt64_next_state(tiny);
    return tinymt64_temper(tiny);
}

#if defined(HAVE_DOUBLE)
/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 *@return random number uniformly distributes in the range [1, 2)
 */
inline static double
tinymt64_double12(tinymt64wp_t * tiny)
{
    ulong x = tinymt64_uint64(tiny);
    x = (x >> 12) ^ tinymt64_double_mask;
    return as_double(x);
}

/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 *@return random number uniformly distributes in the range [0, 1)
 */
inline static double
tinymt64_double01(tinymt64wp_t * tiny)
{
    return tinymt64_double12(tiny) - 1.0;
}
#endif

/**
 * Internal function
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 64-bit integer
 * @return 64-bit integer
 */
inline static ulong
tinymt64_ini_func1(ulong x)
{
    return (x ^ (x >> 59)) * 2173292883993UL;
}

/**
 * Internal function
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 64-bit integer
 * @return 64-bit integer
 */
inline static ulong
tinymt64_ini_func2(ulong x)
{
    return (x ^ (x >> 59)) * 58885565329898161UL;
}

/**
 * Internal function.
 * This function certificate the period of 2^127-1.
 * @param tiny tinymt state vector.
 */
inline static void
tinymt64_period_certification(tinymt64wp_t * tiny)
{
    if ((tiny->s0 & tinymt64_mask) == 0 &&
        tiny->s1 == 0) {
        tiny->s0 = 'T';
        tiny->s1 = 'M';
    }
}

/**
 * This function initializes the internal state array with a 64-bit
 * unsigned integer seed.
 * @param tiny tinymt state vector.
 * @param seed a 64-bit unsigned integer used as a seed.
 */
inline static void
tinymt64_init(tinymt64wp_t * tiny, ulong seed)
{
    ulong status[2];
    status[0] = seed ^ ((ulong)tiny->mat1 << 32);
    status[1] = tiny->mat2 ^ tiny->tmat;
    for (int i = 1; i < TINYMT64_MIN_LOOP; i++) {
        status[i & 1] ^= i + 6364136223846793005UL
            * (status[(i - 1) & 1] ^ (status[(i - 1) & 1] >> 62));
    }
    tiny->s0 = status[0];
    tiny->s1 = status[1];
    tinymt64_period_certification(tiny);
}

/**
 * This function initializes the internal state array,
 * with an array of 64-bit unsigned integers used as seeds
 * @param tiny tinymt state vector.
 * @param init_key the array of 64-bit integers, used as a seed.
 * @param key_length the length of init_key.
 */
inline static void
tinymt64_init_by_array(tinymt64wp_t * tiny,
                       ulong init_key[],
                       int key_length)
{
    const int lag = 1;
    const int mid = 1;
    const int size = 4;
    int i, j;
    int count;
    ulong r;
    ulong st[4];

    st[0] = 0;
    st[1] = tiny->mat1;
    st[2] = tiny->mat2;
    st[3] = tiny->tmat;
    if (key_length + 1 > TINYMT64_MIN_LOOP) {
        count = key_length + 1;
    } else {
        count = TINYMT64_MIN_LOOP;
    }
    r = tinymt64_ini_func1(st[0] ^ st[mid % size]
                           ^ st[(size - 1) % size]);
    st[mid % size] += r;
    r += key_length;
    st[(mid + lag) % size] += r;
    st[0] = r;
    count--;
    for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
        r = tinymt64_ini_func1(st[i % size]
                               ^ st[(i + mid) % size]
                               ^ st[(i + size - 1) % size]);
        st[(i + mid) % size] += r;
        r += init_key[j] + i;
        st[(i + mid + lag) % size] += r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    for (; j < count; j++) {
        r = tinymt64_ini_func1(st[i % size]
                      ^ st[(i + mid) % size]
                      ^ st[(i + size - 1) % size]);
        st[(i + mid) % size] += r;
        r += i;
        st[(i + mid + lag) % size] += r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
        r = tinymt64_ini_func2(st[i % size]
                               + st[(i + mid) % size]
                               + st[(i + size - 1) % size]);
        st[(i + mid) % size] ^= r;
        r -= i;
        st[(i + mid + lag) % size] ^= r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    tiny->s0 = st[0] ^ st[1];
    tiny->s1 = st[2] ^ st[3];
    tinymt64_period_certification(tiny);
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 */
inline static void
tinymt64_status_read(tinymt64wp_t * tiny,
                     __global tinymt64wp_t * g_status)
{
    const size_t id = tinymt_get_sequential_id();
    tiny->s0 = g_status[id].s0;
    tiny->s1 = g_status[id].s1;
    tiny->mat1 = g_status[id].mat1;
    tiny->mat2 = g_status[id].mat2;
    tiny->tmat = g_status[id].tmat;
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 */
inline static void
tinymt64_status_write(__global tinymt64wp_t * g_status,
                      tinymt64wp_t * tiny)
{
    const size_t id = tinymt_get_sequential_id();
    g_status[id].s0 = tiny->s0;
    g_status[id].s1 = tiny->s1;
#if defined(DEBUG)
    g_status[id].mat1 = tiny->mat1;
    g_status[id].mat2 = tiny->mat2;
    g_status[id].tmat = tiny->tmat;
#endif
}

#undef TINYMT64_SHIFT0
#undef TINYMT64_SHIFT1
#undef TINYMT64_MIN_LOOP


#undef KERNEL_PROGRAM

/**
State of tinymt64 RNG.
*/
typedef tinymt64wp_t tinymt64_state;


/**
Generates a random 64-bit unsigned integer using tinymt64 RNG.

@param state State of the RNG to use.
*/
#define tinymt64_ulong(state) tinymt64_uint64(&state)

//#define tinymt64_seed(state, seed) tinymt64_init(state, seed)

/**
Seeds tinymt64 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tinymt64_seed(tinymt64_state* state, ulong seed){
	state->mat1=TINYMT64J_MAT1;
	state->mat2=TINYMT64J_MAT2;
	state->tmat=TINYMT64J_TMAT;
	tinymt64_init(state, seed);
}

/**
Generates a random 32-bit unsigned integer using tinymt64 RNG.

@param state State of the RNG to use.
*/
#define tinymt64_uint(state) ((uint)tinymt64_ulong(state))

/**
Generates a random float using tinymt64 RNG.

@param state State of the RNG to use.
*/
#define tinymt64_float(state) (tinymt64_ulong(state)*TINYMT64_FLOAT_MULTI)

/**
Generates a random double using tinymt64 RNG.

@param state State of the RNG to use.
*/
#define tinymt64_double(state) (tinymt64_ulong(state)*TINYMT64_DOUBLE_MULTI)

/**
Generates a random double using tinymt64 RNG. Since tinymt64 returns 64-bit numbers this is equivalent to tinymt64_double.

@param state State of the RNG to use.
*/
#define tinymt64_double2(state) tinymt64_double(state)


)";
