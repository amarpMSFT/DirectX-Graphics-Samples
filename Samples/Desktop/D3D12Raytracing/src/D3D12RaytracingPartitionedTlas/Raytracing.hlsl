//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// Raytracing.hlsl  (milestone 6: 2 hit groups, shadow rays, reflective donuts)
//
// Shaders:
//   * Raygen              -- primary ray per pixel
//   * Miss                -- primary miss: sets payload to sky/background color
//   * ShadowMiss          -- shadow-ray miss: sets shadowed=false (unshadowed)
//   * ClosestHit_Ball     -- icosphere closest-hit (unit-sphere normal shortcut;
//                            valid for our unit-radius procedural icospheres).
//                            Casts a shadow ray toward the sun.
//   * ClosestHit_Donut    -- torus closest-hit; reads three vertex normals
//                            (one per triangle vertex, packed 3-per-triangle)
//                            from a per-mesh SRV (g_donutVertNormals) and
//                            interpolates them via barycentrics for smooth
//                            shading.  Casts shadow ray AND a reflection
//                            ray; mixes reflected color into the surface
//                            color (reflective donuts).
//
// Per-instance ContributionToHitGroupIndex selects between the two hit groups:
//   0 -> HitGroup_Ball, 1 -> HitGroup_Donut.
// Shader table layout (set by CreateShaderTable() on the CPU):
//   raygen[0] -- Raygen
//   miss  [0] -- Miss        (primary ray miss)
//   miss  [1] -- ShadowMiss  (shadow ray miss)
//   hit   [0] -- HitGroup_Ball
//   hit   [1] -- HitGroup_Donut
// missShaderIndex parameter in TraceRay picks which miss record runs.
//
// Recursion model: MaxRecursionDepth = 2 in the pipeline state.
//   depth 0: raygen
//   depth 1: primary ray hit -> shadow/reflection rays
//   depth 2: reflection ray's hit -> shadow ray only (no further reflection;
//            enforced via payload.depth check inside ClosestHit_Donut).
//

#define HLSL
#include "RaytracingHlslCompat.h"

RaytracingAccelerationStructure g_tlas             : register(t0);
ByteAddressBuffer               g_donutVertNormals : register(t1);   // 3 float3 per triangle, IB-order
RWTexture2D<float4>             g_output           : register(u0);
ConstantBuffer<SceneConstantBuffer> g_scene        : register(b0);

// ---------- Payloads ----------
//
// Payload (primary + reflection rays) carries a colour + recursion depth.
// The depth prevents the reflection ray from launching another reflection
// (would exceed MaxRecursionDepth = 2).
struct [raypayload] Payload
{
    float3 colour : read(caller) : write(caller, closesthit, miss);
    uint   depth  : read(closesthit) : write(caller);
};

// ShadowPayload is a 1-byte signal: did the shadow ray hit anything?
// Default-initialised to "shadowed = true" before TraceRay; the
// ShadowMiss shader clears it on miss.  With
// RAY_FLAG_SKIP_CLOSEST_HIT_SHADER + ACCEPT_FIRST_HIT_AND_END_SEARCH no
// closest-hit shader runs for the shadow ray; only the miss can mutate
// the payload.
struct [raypayload] ShadowPayload
{
    uint shadowed : read(caller) : write(caller, miss);
};

// ---------- Primary ray construction ----------

RayDesc PrimaryRayFromPixel(uint2 px, uint2 dim)
{
    float2 xy = (float2(px) + 0.5) / float2(dim) * 2.0 - 1.0;
    xy.y = -xy.y;
    float4 world = mul(float4(xy, 0, 1), g_scene.projectionToWorld);
    world.xyz /= world.w;
    RayDesc ray;
    ray.Origin    = g_scene.cameraOriginAs.xyz;
    ray.Direction = normalize(world.xyz - ray.Origin);
    ray.TMin      = 0.001;
    ray.TMax      = 10000.0;
    return ray;
}

// ---------- Shadow helper ----------
//
// Traces a shadow ray from `worldOrigin` toward `sunDir`.  Returns 0.0
// if the path is blocked (point in shadow) or 1.0 if it's clear.
//
// We OFFSET the ray origin by a small distance along `normal` to avoid
// self-intersection with the surface we're shading.  TMin=0 + offset is
// the standard recipe; using TMin>0 alone is less robust because the
// surface might be at distance < TMin from the (un-offset) origin.
float ShadowVisibility(float3 worldOrigin, float3 normal, float3 sunDir)
{
    RayDesc s;
    s.Origin    = worldOrigin + normal * 0.001;
    s.Direction = sunDir;
    s.TMin      = 0.0;
    s.TMax      = 1000.0;
    ShadowPayload sp;
    sp.shadowed = 1;
    TraceRay(g_tlas,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             /*InstanceInclusionMask*/0xFF,
             /*RayContributionToHitGroupIndex*/0,
             /*MultiplierForGeometryContributionToHitGroupIndex*/0,
             /*MissShaderIndex*/1,
             s, sp);
    return sp.shadowed ? 0.0 : 1.0;
}

// ---------- Raygen ----------

[shader("raygeneration")]
void Raygen()
{
    uint2 px  = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;
    RayDesc ray = PrimaryRayFromPixel(px, dim);
    Payload pay;
    pay.colour = g_scene.missColorAndTime.rgb;
    pay.depth  = 0;
    TraceRay(g_tlas,
             RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
             /*InstanceInclusionMask*/0xFF,
             /*RayContributionToHitGroupIndex*/0,
             /*MultiplierForGeometryContributionToHitGroupIndex*/0,
             /*MissShaderIndex*/0,
             ray, pay);
    g_output[px] = float4(pay.colour, 1.0);
}

// ---------- Miss shaders ----------

[shader("miss")]
void Miss(inout Payload pay)
{
    // Soft gradient based on ray Y so reflections show "sky vs ground" hint.
    float3 d = normalize(WorldRayDirection());
    float upMix = saturate(d.y * 0.5 + 0.5);
    float3 sky    = float3(0.04, 0.08, 0.12);    // upper hemisphere
    float3 ground = float3(0.03, 0.02, 0.02);    // lower hemisphere
    pay.colour = lerp(ground, sky, upMix);
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload pay)
{
    pay.shadowed = 0;
}

// ---------- Closest-hit shaders ----------

[shader("closesthit")]
void ClosestHit_Ball(inout Payload pay, in BuiltInTriangleIntersectionAttributes attr)
{
    // Unit-sphere normal shortcut (valid for our unit icospheres).
    float3 worldHit = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 objHit   = mul(WorldToObject3x4(), float4(worldHit, 1));
    float3 nObj     = normalize(objHit);
    float3 nWorld   = normalize(mul((float3x3)ObjectToWorld3x4(), nObj));
    if (dot(nWorld, WorldRayDirection()) > 0) nWorld = -nWorld;

    float3 sunDir = normalize(g_scene.lightDirAndPad.xyz);
    // Shadow rays only on PRIMARY hits.  Bounce hits (pay.depth > 0) would
    // need MaxRecursionDepth >= 3; we cap at 2.  Reflected/secondary hits
    // see the surface fully lit -- visually acceptable for the demo.
    float  vis    = (pay.depth == 0) ? ShadowVisibility(worldHit, nWorld, sunDir) : 1.0;
    float  ndotl  = saturate(dot(nWorld, sunDir));

    float3 base = 0.5 + 0.5 * nWorld;       // normal-as-tint
    pay.colour  = base * (0.18 + 0.82 * ndotl * vis);
}

[shader("closesthit")]
void ClosestHit_Donut(inout Payload pay, in BuiltInTriangleIntersectionAttributes attr)
{
    // Smooth normal: read the THREE per-vertex normals for this triangle
    // (packed 3-per-triangle in g_donutVertNormals by
    // MeshAssets::BuildPerTriVertexNormalsBuffer) and interpolate them via
    // the barycentric attributes.  Same pattern as the clustered sample's
    // LoadHitContext at Raytracing.hlsl:550-557.
    //
    // attr.barycentrics: x = weight of vertex 1, y = weight of vertex 2.
    // Vertex 0's weight = 1 - x - y.  Order matches IB[primIdx*3 + 0/1/2].
    uint primIdx = PrimitiveIndex();
    uint base    = primIdx * 36;     // 3 normals * 12 bytes
    float3 n0 = asfloat(g_donutVertNormals.Load3(base +  0));
    float3 n1 = asfloat(g_donutVertNormals.Load3(base + 12));
    float3 n2 = asfloat(g_donutVertNormals.Load3(base + 24));
    float  bw1 = attr.barycentrics.x;
    float  bw2 = attr.barycentrics.y;
    float  bw0 = 1.0 - bw1 - bw2;
    float3 nObj   = normalize(n0 * bw0 + n1 * bw1 + n2 * bw2);
    float3 nWorld = normalize(mul((float3x3)ObjectToWorld3x4(), nObj));
    if (dot(nWorld, WorldRayDirection()) > 0) nWorld = -nWorld;

    float3 worldHit = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 sunDir   = normalize(g_scene.lightDirAndPad.xyz);

    // Shadow (primary hits only -- see ClosestHit_Ball comment).
    float vis   = (pay.depth == 0) ? ShadowVisibility(worldHit, nWorld, sunDir) : 1.0;
    float ndotl = saturate(dot(nWorld, sunDir));

    // Reflection: only at recursion depth 0 (primary hits).  A reflected
    // ray hitting a donut just shades it without further recursion.
    float3 reflectColor = float3(0, 0, 0);
    if (pay.depth < 1)
    {
        RayDesc r;
        r.Origin    = worldHit + nWorld * 0.001;
        r.Direction = reflect(WorldRayDirection(), nWorld);
        r.TMin      = 0.0;
        r.TMax      = 1000.0;
        Payload p2;
        p2.colour = g_scene.missColorAndTime.rgb;
        p2.depth  = pay.depth + 1;
        TraceRay(g_tlas,
                 RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                 0xFF, 0, 0, 0, r, p2);
        reflectColor = p2.colour;
    }

    // Donut surface color: warm metallic-ish tint.
    float3 base_col = float3(0.65, 0.45, 0.20);
    float3 surface  = base_col * (0.12 + 0.88 * ndotl * vis);

    // 55% reflective; brighter on rim (Fresnel-ish).
    float fres = pow(1.0 - saturate(dot(-WorldRayDirection(), nWorld)), 2.0);
    float refl = 0.45 + 0.45 * fres;

    pay.colour = lerp(surface, reflectColor, refl);
}
