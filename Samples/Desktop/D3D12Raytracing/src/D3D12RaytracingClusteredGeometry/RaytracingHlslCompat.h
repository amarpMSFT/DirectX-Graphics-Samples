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
    XMFLOAT4  miscParams;        // .x = aspect ratio, .y = tan(fov/2), .z/.w = unused
    XMFLOAT4  lightDir;          // .xyz = world-space direction TO the sun (normalized),
                                 // .w   = ambient floor [0..1] (so shadowed regions remain
                                 //        readable instead of pitch black)
};

// Material kinds (one per scene instance).
//   OPAQUE:       diffuse + shadow only. Instance gets FORCE_OPAQUE flag, so
//                 the any-hit shader is never invoked - cheapest path.
//   REFLECTIVE:   diffuse + shadow PLUS one-bounce reflection. Mixed by
//                 reflectivity.
//   REFRACTIVE:   diffuse + shadow PLUS one-bounce refraction (Snell's law,
//                 IOR per material). Mixed by translucency. Animated sphere
//                 uses this to morph the distortion of what's behind it.
//   STOCHASTIC:   diffuse + shadow but the any-hit shader probabilistically
//                 rejects per triangle (probability = translucency), giving
//                 a noisy "frosted-glass" partial transparency. NOT marked
//                 FORCE_OPAQUE on the instance.
#define MAT_KIND_OPAQUE      0u
#define MAT_KIND_REFLECTIVE  1u
#define MAT_KIND_REFRACTIVE  2u
#define MAT_KIND_STOCHASTIC  3u

#define NUM_MATERIAL_SLOTS   9u   // 4 spheres + torus + cube + floor + animated + klein

struct MaterialDesc
{
    // baseColor.xyz multiplies the shader's per-cluster colour. For our demo
    // we always set it to (1,1,1) so the cluster-rainbow shows through every
    // material. baseColor.w is unused.
    XMFLOAT4 baseColor;
    // params.x = reflectivity in [0..1]   (REFLECTIVE only)
    // params.y = translucency in [0..1]   (REFRACTIVE: blend of refract vs
    //                                       surface; STOCHASTIC: any-hit
    //                                       reject probability)
    // params.z = index of refraction       (REFRACTIVE only; e.g. 1.5 = glass)
    // params.w = unused
    XMFLOAT4 params;
    // .x = MAT_KIND_*; the rest is padding so the struct is 16-byte-aligned
    // for both HLSL StructuredBuffer reads and CPU upload.
    uint     kind;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};
