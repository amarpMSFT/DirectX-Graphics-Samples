// Raytracing.hlsl -- MINIMAL COMPRESSED1 BUG REPRO SHADER
//
// One primary ray per pixel.  Hit returns a per-(cluster, primitive,
// instance) hash colour so each surviving triangle is visibly distinct;
// miss returns sky blue.  No any-hit / shadow / reflection / refraction
// shaders -- the bug demonstration only needs BVH traversal of the
// CLAS+BLAS chain to fail, not any of the optical effects the original
// sample was showcasing.

#define HLSL
#include "RaytracingHlslCompat.h"

RaytracingAccelerationStructure   Scene       : register(t0);
RWTexture2D<float4>               Output      : register(u0);
ConstantBuffer<SceneConstantBuffer> g_scene   : register(b0);

struct [raypayload] Payload {
    float4 color : write(caller, closesthit, miss) : read(caller);
};
typedef BuiltInTriangleIntersectionAttributes Attribs;

[shader("raygeneration")]
void RayGen()
{
    const uint2  px      = DispatchRaysIndex().xy;
    const uint2  dim     = DispatchRaysDimensions().xy;
    const float2 ndc     = (float2(px) + 0.5) / float2(dim) * 2.0 - 1.0;
    const float2 view2d  = float2(ndc.x, -ndc.y);
    const float  aspect      = g_scene.miscParams.x;
    const float  tanHalfFov  = g_scene.miscParams.y;
    const float3 dirView     = normalize(float3(view2d.x * aspect * tanHalfFov,
                                                view2d.y * tanHalfFov,
                                                1.0));
    const float3 dirWorld    = mul((float3x3)g_scene.viewToWorld, dirView);

    RayDesc r;
    r.Origin    = g_scene.cameraPosition.xyz;
    r.Direction = normalize(dirWorld);
    r.TMin      = 0.001;
    r.TMax      = 1000.0;

    Payload p; p.color = float4(0, 0, 0, 1);

    TraceRay(Scene,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        /*InstanceInclusionMask*/0xff,
        /*RayContributionToHitGroupIndex*/0,
        /*MultiplierForGeometryContributionToHitGroupIndex*/0,
        /*MissShaderIndex*/0,
        r, p);
    Output[px] = p.color;
}

[shader("miss")]
void Miss(inout Payload p) { p.color = float4(0.40, 0.60, 0.90, 1); }

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
