//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// RaytracingHlslCompat.h
//
// CPU-side and HLSL-side share this header to keep struct layouts in sync.
// Members must be 16-byte aligned and use fundamentals that exist in both
// languages.
//
#pragma once

#ifdef HLSL
    // HLSL side: provide aliases for the C++ math types this header uses.
    typedef float3   XMFLOAT3;
    typedef float4   XMFLOAT4;
    typedef float4x4 XMFLOAT4X4;
    typedef uint     UINT;
#else
    // C++ side: pull in DirectXMath for the matching types.
    #include <DirectXMath.h>
    using XMFLOAT3   = DirectX::XMFLOAT3;
    using XMFLOAT4   = DirectX::XMFLOAT4;
    using XMFLOAT4X4 = DirectX::XMFLOAT4X4;
#endif

// Global root signature slot map.  Kept compact for the skeleton; will grow
// as cluster metadata / per-instance material lookups come in.
#define PT_GRS_OutputUavSlot              0
#define PT_GRS_AccelerationStructureSlot  1
#define PT_GRS_SceneCBVSlot               2
#define PT_GRS_DonutVertNormalsSrvSlot    3   // milestone 6: per-mesh SRV for donut face normals

// Per-frame constants for raygen / closest-hit.
//
// `cameraOriginAs` is the camera position in **acceleration-structure-space**
// (i.e. world position MINUS the per-frame `asOrigin` that all partition
// translations are biased against -- see docs/design.md).  In milestone 2a
// asOrigin is just (0,0,0) so cameraOriginAs == cameraWorld.
struct SceneConstantBuffer
{
    XMFLOAT4X4 projectionToWorld;     // inverse(view*proj), used by raygen
    XMFLOAT4   cameraOriginAs;        // .xyz = camera position in AS-space, .w = 1
    XMFLOAT4   asOriginWorld;         // .xyz = AS-origin world position, .w unused
    XMFLOAT4   lightDirAndPad;        // .xyz = sun dir (toward light), normalized
    XMFLOAT4   missColorAndTime;      // .rgb = miss colour, .a = seconds since start
};
