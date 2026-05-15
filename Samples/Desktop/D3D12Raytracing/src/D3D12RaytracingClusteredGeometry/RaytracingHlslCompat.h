//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// RaytracingHlslCompat.h
//
// Constant-buffer / shader-record types shared between C++ and HLSL.
// (HlslCompat.h handles the HLSL-side typedef'ing of XMFLOAT3/etc to float3 etc.)
//
#pragma once

#ifdef HLSL
#include "HlslCompat.h"
#else
#include <DirectXMath.h>
typedef DirectX::XMFLOAT4   XMFLOAT4;
typedef DirectX::XMMATRIX   XMMATRIX;
typedef UINT                uint;
#endif

struct SceneConstantBuffer
{
    XMMATRIX  viewToWorld;       // ray gen camera basis (also applies translation)
    XMFLOAT4  cameraPosition;    // .xyz = world-space eye position
    XMFLOAT4  miscParams;        // .x = aspect ratio, .y = tan(fov/2), .z = unused, .w = unused
};
