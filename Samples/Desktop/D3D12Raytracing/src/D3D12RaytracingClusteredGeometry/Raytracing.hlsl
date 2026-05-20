// Raytracing.hlsl -- MINIMAL COMPRESSED1 BUG REPRO SHADER
//
// Stripped down to the bare minimum needed to demonstrate the NVIDIA
// COMPRESSED1 cluster-vertex-format rendering corruption: 1 primary
// ray per pixel, hit returns a per-primitive hash colour, miss
// returns sky blue.  No materials, no normals, no shadows, no
// reflections, no refractions, no any-hit logic.  The HIT-GROUP
// signatures (OpaqueHit / GlassHit / GlassAnyHit / ShadowMiss) are
// kept as empty stubs so the existing 8-record hit-group table
// layout still binds without modification.  Real corruption only
// needs the closesthit + traversal -- everything else is noise.

#define HLSL
#include "RaytracingHlslCompat.h"

RaytracingAccelerationStructure   Scene       : register(t0);
RWTexture2D<float4>               Output      : register(u0);
ConstantBuffer<SceneConstantBuffer> g_scene   : register(b0);

struct [raypayload] Payload {
    float4 color   : write(caller, closesthit, miss) : read(caller);
    uint   depth   : write(caller, closesthit, miss) : read(caller, closesthit);
    uint   inGlass : write(caller, closesthit, miss) : read(caller, closesthit);
};

struct [raypayload] ShadowPayload { bool inShadow : write(caller, miss) : read(caller); };
typedef BuiltInTriangleIntersectionAttributes Attribs;

// =========================================================================
// Raygen: 1 primary ray / pixel, sampled at pixel center, no AA, no time
// advancing.  Camera basis comes from g_scene.viewToWorld + cameraPosition;
// miscParams.x = aspect, miscParams.y = tan(fov/2).
// =========================================================================
[shader("raygeneration")]
void RayGen()
{
    const uint2  px      = DispatchRaysIndex().xy;
    const uint2  dim     = DispatchRaysDimensions().xy;
    const float2 ndc     = (float2(px) + 0.5) / float2(dim) * 2.0 - 1.0;
    // y flipped so +y is up in world after the viewToWorld transform.
    const float2 view2d  = float2(ndc.x, -ndc.y);
    // Aspect + half-FOV from miscParams.
    const float aspect      = g_scene.miscParams.x;
    const float tanHalfFov  = g_scene.miscParams.y;
    const float3 dirView    = normalize(float3(view2d.x * aspect * tanHalfFov,
                                               view2d.y * tanHalfFov,
                                               1.0));
    const float3 dirWorld   = mul((float3x3)g_scene.viewToWorld, dirView);

    RayDesc r;
    r.Origin    = g_scene.cameraPosition.xyz;
    r.Direction = normalize(dirWorld);
    r.TMin      = 0.001;
    r.TMax      = 1000.0;

    Payload p = (Payload)0;
    p.color = float4(0, 0, 0, 1);

    TraceRay(Scene,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        /*InstanceInclusionMask*/0xff,
        /*RayContributionToHitGroupIndex*/0,
        /*MultiplierForGeometryContributionToHitGroupIndex*/2,
        /*MissShaderIndex*/0,
        r, p);
    Output[px] = p.color;
}

// Sky-blue miss; primary rays that miss return this.
[shader("miss")]
void Miss(inout Payload p) { p.color = float4(0.40, 0.60, 0.90, 1); }
[shader("miss")]
void ShadowMiss(inout ShadowPayload sp) { sp.inShadow = false; }

// =========================================================================
// Hit shaders: BOTH OpaqueHit and GlassHit return a per-primitive hash
// colour so each cube face / cluster face appears as a distinct shade.
// This makes "missing face" corruption immediately visible: a healthy
// cube shows 6 differently-shaded faces (3 visible at a time); a corrupt
// cube under COMPRESSED1 shows only the few faces that survived BVH
// traversal.
// =========================================================================
[shader("closesthit")]
void OpaqueHit(inout Payload p, in Attribs a)
{
    const uint cid     = ClusterID();
    const uint pid     = PrimitiveIndex();
    const uint inst    = InstanceIndex();
    const uint h       = (cid * 2654435761u) ^ (pid * 374761393u) ^ (inst * 668265263u);
    p.color = float4(((h >>  0) & 0xFF) / 255.0,
                     ((h >>  8) & 0xFF) / 255.0,
                     ((h >> 16) & 0xFF) / 255.0,
                     1.0);
}

[shader("closesthit")]
void GlassHit(inout Payload p, in Attribs a) { OpaqueHit(p, a); }

[shader("anyhit")]
void GlassAnyHit(inout Payload p, in Attribs a) { /* accept all */ }
