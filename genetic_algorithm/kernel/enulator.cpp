
#include "kernelCppHidings.hpp"

namespace artificial_neural_network::openClEmulator
{

std::vector<size_t> global_id;       // global index
std::vector<size_t> global_size;     // global range
std::vector<size_t> local_id;        // local index within group
std::vector<size_t> local_size;      // group size
std::vector<size_t> num_groups;      // number of groups
std::vector<size_t> group_id;        // group ID
std::vector<size_t> global_offset;   // global offset

};  // namespace artificial_neural_network::openClEmulator
