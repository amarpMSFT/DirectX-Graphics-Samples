//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

// Fast-path permutation of CalculateMeanVarianceCS that exchanges each row's values across
// wave lanes (WaveReadLaneAt) instead of through groupshared memory. It assumes waves of >= 16
// lanes AND that threads are packed to lanes in row-major SV_GroupIndex order, neither of which
// is guaranteed by HLSL. The host (RTAOGpuKernels::CalculateMeanVariance) only selects this
// permutation when the device reports WaveLaneCountMin >= 16; otherwise the portable default
// (CalculateMeanVarianceCS.hlsl, groupshared exchange) is used.
#define RTAO_WAVE_READ_LANE_PATH
#include "CalculateMeanVarianceCS.hlsl"
