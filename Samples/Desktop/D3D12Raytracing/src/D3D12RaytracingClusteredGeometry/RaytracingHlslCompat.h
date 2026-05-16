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

// Per-instance material.  Each instance carries an independent BLEND of
// optical effects rather than a single "kind" - so a chrome sphere with
// frosted holes punched through it is just `reflectivity=0.85,
// translucency=0.40` on the same MaterialDesc.  Closesthit composes the
// final colour as:
//
//   base = baseColor * cluster_palette                 (per-cluster tint)
//   surface = base * (ambient + (1-ambient)*NdotL*shadow)
//
//   if (refractivity > 0 && depth <= 1)
//       refracted = TraceBounce(refracted_dir, depth+1)
//       surface   = lerp(surface, refracted, refractivity)
//
//   if (reflectivity > 0 && depth == 0)
//       reflected = TraceBounce(reflected_dir, depth+1)
//       surface   = lerp(surface, reflected, reflectivity)
//
// AnyHit fires for any instance whose translucency > 0 OR whose
// refractivity > 0 (those instances do NOT get FORCE_OPAQUE).  AnyHit
// only does the stochastic reject - refraction is handled in closesthit
// via the bounce ray.
//   - translucency  -> per-triangle hash; IgnoreHit() if random <
//                      translucency.  Gives "frosted-glass" speckle.
//   - refractivity  -> AnyHit accepts unconditionally; closesthit's
//                      Snell-bend ray delivers the see-through.
//
// Per-instance FORCE_OPAQUE flag (and per-cluster OPAQUE flag) is set
// when BOTH translucency == 0 AND refractivity == 0 - opaque materials
// (with or without reflection) skip any-hit dispatch entirely.
#define NUM_MATERIAL_SLOTS   9u   // 4 spheres + torus + cube + floor + animated + klein

struct MaterialDesc
{
    // baseColor.xyz multiplies the shader's per-cluster colour.  Use
    // (1,1,1) to let the cluster-rainbow show through unmodified, or a
    // tint to colour-grade the whole instance.  baseColor.w is unused.
    XMFLOAT4 baseColor;
    float    reflectivity;   // [0..1] mirror-reflection blend at depth 0
    float    refractivity;   // [0..1] Snell-refraction blend at depth 0+1
    float    ior;            // refractive index (1.5=glass, 1.33=water, ignored if refractivity==0)
    float    translucency;   // [0..1] stochastic any-hit reject probability
};
