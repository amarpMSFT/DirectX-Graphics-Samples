//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// Raytracing.hlsl  (skeleton -- one raygen + one closest-hit + one miss)
//
// Milestone 2a render: primary visibility only, shade by triangle face
// normal lit by a single hard-coded sun direction.  No reflections, no
// shadow rays, no per-instance materials.  Just enough to verify that the
// (P)TLAS is producing hits.
//

#define HLSL
#include "RaytracingHlslCompat.h"

RaytracingAccelerationStructure g_tlas      : register(t0);
RWTexture2D<float4>             g_output    : register(u0);
ConstantBuffer<SceneConstantBuffer> g_scene : register(b0);

// ---------- Hit attributes / payload ----------

struct [raypayload] Payload
{
    float3 colour : read(caller) : write(caller, closesthit, miss);
    float  hitT   : read(caller) : write(caller, closesthit, miss);  // -1 on miss
};

// ---------- Primary ray construction ----------

// Build a ray from the pixel center in AS-space.  cameraOriginAs.xyz is
// already in AS-space; the direction comes from unprojecting the
// pixel-center NDC through projectionToWorld and subtracting cameraOrigin.
// (projectionToWorld is built CPU-side from the same AS-space camera, so
// the unprojected point is also in AS-space.)
RayDesc PrimaryRayFromPixel(uint2 px, uint2 dim)
{
    // Pixel center to NDC [-1, +1].  Flip y so up=+y in NDC.
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

// ---------- Raygen ----------

[shader("raygeneration")]
void Raygen()
{
    uint2 px  = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    RayDesc ray = PrimaryRayFromPixel(px, dim);

    Payload pay;
    pay.colour = g_scene.missColorAndTime.rgb;
    pay.hitT   = -1.0;

    TraceRay(g_tlas,
             RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
             /*InstanceInclusionMask*/0xFF,
             /*RayContributionToHitGroupIndex*/0,
             /*MultiplierForGeometryContributionToHitGroupIndex*/0,
             /*MissShaderIndex*/0,
             ray, pay);

    g_output[px] = float4(pay.colour, 1.0);
}

// ---------- Miss ----------

[shader("miss")]
void Miss(inout Payload pay)
{
    pay.colour = g_scene.missColorAndTime.rgb;
    pay.hitT   = -1.0;
}

// ---------- Closest hit ----------
//
// Milestone 2a shortcut: derive the object-space normal as the normalized
// object-space hit position.  This is exact for unit spheres centered at
// object origin (which is all we have in 2a).  Phase 2b switches to a
// general path:
//   * cluster path: per-cluster vertex-normal table indexed by ClusterID()
//                   + primitive index (same pattern as the clustered sample)
//   * traditional path: per-instance VB/IB SRV lookup using PrimitiveIndex()

[shader("closesthit")]
void ClosestHit(inout Payload pay, in BuiltInTriangleIntersectionAttributes attr)
{
    // World hit position from the ray.
    float3 worldHit = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    // Object-space hit (3x4 row-major matrix from world to object space).
    float3 objHit = mul(WorldToObject3x4(), float4(worldHit, 1));

    // Unit-sphere normal in object space.
    float3 nObj   = normalize(objHit);

    // To world space via the 3x3 upper block of ObjectToWorld3x4().
    // For uniform-scale rigid transforms this is correct without an
    // explicit inverse-transpose.
    float3 nWorld = normalize(mul((float3x3)ObjectToWorld3x4(), nObj));

    // Flip if facing away from the ray (defensive; back-face culling on).
    if (dot(nWorld, WorldRayDirection()) > 0) nWorld = -nWorld;

    // Simple Lambert + low ambient + normal-as-tint.
    float ndotl = saturate(dot(nWorld, normalize(g_scene.lightDirAndPad.xyz)));
    float3 base = 0.5 + 0.5 * nWorld;
    pay.colour  = base * (0.18 + 0.82 * ndotl);
    pay.hitT    = RayTCurrent();
}
