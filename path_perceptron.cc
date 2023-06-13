/*
 * Copyright (c) 2004-2006 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/pred/path_perceptron.hh"


#include <math.h>
#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/Fetch.hh"

namespace gem5
{

namespace branch_prediction
{

PathPerceptron::PathPerceptron(const PathPerceptronParams &params)
    : BPredUnit(params),
      globalHistoryLength(params.globalHistoryLength),
      localPredictorSize(params.localPredictorSize),
      globalHistory(params.numThreads, 0),
      specGlobalHistory(params.numThreads, 0),
      localWeightBits(8),
      numLocalWeights(globalHistoryLength + 1),
      numPerceptrons(localPredictorSize / localWeightBits / numLocalWeights),
      weights(numPerceptrons, std::vector<int8_t>(numLocalWeights, 0)),
      indexMask(numPerceptrons - 1)
{
    if (!isPowerOf2(localPredictorSize)) {
        fatal("Invalid local predictor size!\n");
    }

    if (!isPowerOf2(numPerceptrons)) {
        fatal("Invalid number of perceptrons! Check globalHistoryLength must be 2^n - 1.\n");
    }

    if (globalHistoryLength == 63)
	    globalHistoryMask = UINT64_MAX;
    else
    	globalHistoryMask = (1ULL << (globalHistoryLength + 1)) - 1ULL;
    theta = (int)floor(2.14 * (globalHistoryLength + 1) + 20.58);

    // For saturated weight update
    maxWeight = INT8_MAX;
    minWeight = INT8_MIN;

    runningSum.assign(globalHistoryLength + 1, 0);
    specRunningSum.assign(globalHistoryLength + 1, 0);

    branchPath.assign(params.numThreads, std::vector<Addr>());

    DPRINTF(Fetch, "index mask: %#x\n", indexMask);

    DPRINTF(Fetch, "local predictor size: %i\n",
            localPredictorSize);

    DPRINTF(Fetch, "local weight bits: %i\n", localWeightBits);

    DPRINTF(Fetch, "instruction shift amount: %i\n",
            instShiftAmt);
    
    DPRINTF(Fetch, "global history mask: %#x\n", globalHistoryMask);

    DPRINTF(Fetch, "theta: %i\n", theta);
}

void PathPerceptron::btbUpdate(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    // Place holder for a function that is called to update predictor history when
    // a BTB entry is invalid or not found.
    globalHistory[tid] &= (globalHistoryMask & ~1ULL);
}


void PathPerceptron::updateBranchPath(ThreadID tid, Addr branch_addr)
{
    if (branchPath[tid].size() >= globalHistoryLength + 1)
        branchPath[tid].pop_back();
    branchPath[tid].insert(branchPath[tid].begin(), branch_addr);
}


int8_t PathPerceptron::updateWeight(int8_t weight, bool taken)
{
    // weight hasn't been +- 1
    if (taken && weight < maxWeight)
        return weight + 1;
    if (!taken && weight > minWeight)
        return weight - 1;
    return weight;
}


bool
PathPerceptron::lookup(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    bool taken;
    unsigned perceptron_idx = getLocalIndex(branch_addr);

    DPRINTF(Fetch, "Looking up index %#x\n",
            perceptron_idx);

    int y_out = specRunningSum[globalHistoryLength] + weights[perceptron_idx][0];
    taken = (y_out >= 0) ? true : false;

    DPRINTF(Fetch, "prediction is %i.\n", y_out);

    updateBranchPath(tid, branch_addr);
    
    // Create bp_history
    BPHistory *history = new BPHistory;
    history->globalHistory   = specGlobalHistory[tid];
    history->globalPredTaken = taken;
    bp_history = (void *)history;

    int k_j;
    std::vector<int> specRunningSum_prime(globalHistoryLength + 1, 0);
    for (int j = 1; j <= globalHistoryLength; j++)
    {
        k_j = globalHistoryLength - j;
        if (taken)
            specRunningSum_prime[k_j + 1] = specRunningSum[k_j] + weights[perceptron_idx][j];
        else
            specRunningSum_prime[k_j + 1] = specRunningSum[k_j] - weights[perceptron_idx][j];
    }
    specRunningSum = specRunningSum_prime;
    specRunningSum[0] = 0;

    specGlobalHistory[tid] = ((specGlobalHistory[tid] << 1) | taken);
    specGlobalHistory[tid] = (specGlobalHistory[tid] & globalHistoryMask);

    return taken;
}

void
PathPerceptron::update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history,
                bool squashed, const StaticInstPtr & inst, Addr corrTarget)
{
    assert(bp_history);
    unsigned perceptron_idx = getLocalIndex(branch_addr);
    int y_out = specRunningSum[globalHistoryLength] + weights[perceptron_idx][0];
    unsigned long long spec_history = specGlobalHistory[tid]; // global history before update to find correlation

    // Update non-speculative global history
    globalHistory[tid] = ((globalHistory[tid] << 1) | taken);
    globalHistory[tid] = (globalHistory[tid] & globalHistoryMask);

    // Update non-speculative running sum
    std::vector<int> runningSum_prime(globalHistoryLength + 1, 0);
    for (int j = 1; j <= globalHistoryLength; j++)
    {
        int k_j = globalHistoryLength - j;
        if (taken)
            runningSum_prime[k_j + 1] = runningSum[k_j] + weights[perceptron_idx][j];
        else
            runningSum_prime[k_j + 1] = runningSum[k_j] - weights[perceptron_idx][j];
    }
    runningSum = runningSum_prime;
    runningSum[0] = 0;

    if (squashed || abs(y_out) <= theta) {
        if (squashed)
        {
            specGlobalHistory[tid] = globalHistory[tid];
            specRunningSum = runningSum;
        }
        weights[perceptron_idx][0] = updateWeight(weights[perceptron_idx][0], taken);
        
        for (int j = 1; j <= globalHistoryLength; j++)
        {
            // Use mod in case not enough branch history
            unsigned k = (getLocalIndex(branchPath[tid][j % branchPath.size()]));    // branchPath[1 ... h]
            bool correlation = (((spec_history >> j) & 1ULL) == taken);
            weights[k][j] = updateWeight(weights[k][j], correlation);
        }
    }

    DPRINTF(Fetch, "Looking up index %#x\n", perceptron_idx);
}

void PathPerceptron::squash(ThreadID tid, void *bp_history)
{
    BPHistory *history = static_cast<BPHistory *>(bp_history);
	specGlobalHistory[tid] = globalHistory[tid];
    specRunningSum = runningSum;
    delete history;
}

// TODO: Can change to consider XOR?
inline
unsigned long long
PathPerceptron::getLocalIndex(Addr &branch_addr)
{
	// unsigned long long global_segment = (specGlobalHistory[tid] & indexMask);
	// unsigned long long addr_segment = ((branch_addr >> instShiftAmt) & indexMask);
	// return (global_segment ^ addr_segment);
    return ((branch_addr >> instShiftAmt) & indexMask);
}

void
PathPerceptron::uncondBranch(ThreadID tid, Addr pc, void *&bp_history)
{
	// Create BPHistory and pass it back to be recorded.
  	BPHistory *history = new BPHistory;
  	history->globalHistory = specGlobalHistory[tid];
  	history->globalPredTaken = true;
  	history->globalUsed = true;
  	bp_history = static_cast<void *>(history);
    
	updateBranchPath(tid, pc);
    specGlobalHistory[tid] = ((specGlobalHistory[tid] << 1) | 1);
    specGlobalHistory[tid] = (specGlobalHistory[tid] & globalHistoryMask);
}

} // namespace branch_prediction
} // namespace gem5
