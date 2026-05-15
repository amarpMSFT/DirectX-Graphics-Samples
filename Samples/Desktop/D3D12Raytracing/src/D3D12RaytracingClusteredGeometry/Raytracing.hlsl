// Raytracing.hlsl
//
// Renders the clustered scene with a closest-hit shader that returns a colour
// derived from ClusterID(). Single primary ray per pixel.
//
// Pipeline is created with D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_CLUSTERED_GEOMETRY
// so ClusterID() returns the actual cluster id on hits into a Cluster BLAS
// (and CLUSTER_ID_INVALID = 0xFFFFFFFF on hits into a classic triangle BLAS).

#define HLSL
#include "RaytracingHlslCompat.h"

RaytracingAccelerationStructure Scene  : register(t0);
RWTexture2D<float4>             Output : register(u0);
ConstantBuffer<SceneConstantBuffer> g_scene : register(b0);

struct [raypayload] Payload
{
    float4 color : write(miss, closesthit) : read(caller);
};
struct Attribs { float2 bary; };

// ============================================================================
// WARP-WORKAROUND  (DELETE WHEN WARP IS FIXED)
// ----------------------------------------------------------------------------
// The natural way to spell a per-cluster colour palette is:
//
//     static const float3 kPalette[16] = { float3(...), float3(...), ... };
//     ...
//     base = kPalette[cid % 16];
//
// That faults the experimental WARP build (d3d10warp.dll 1.0.19.0.261613-0416
// as of 2026-05-13) in CreateStateObject:
//
//   d3d10warp!CShaderInfo::CompileShaderDXILLibrary  (shaderinfo.cpp:2503)
//   Assert(pArrayElementType->isFloatTy() || pArrayElementType->isIntegerTy()
//          || bIs64BitType || bIs16BitType)
//
// The DXIL global is `[16 x <3 x float>]` and WARP's array-of-immediate-
// constants pass only accepts SCALAR array element types. The assert is
// missing a `FixedVectorTyID` case. NVIDIA's driver compiles the same DXIL
// without complaint.
//
// Workaround: compute the palette colour algebraically with a cosine-palette
// (Inigo Quilez style). Same 16-step rainbow visual outcome, no static-const
// array-of-vec lookup. Once WARP picks up the fix, this function can be
// replaced with the natural array indexing.
// ============================================================================
float3 ClusterColor(uint cid)
{
    float t = float(cid % 16) * (1.0 / 16.0);             // 0..1 around hue wheel
    float3 a = float3(0.55, 0.55, 0.55);
    float3 b = float3(0.45, 0.45, 0.45);
    float3 c = float3(1.00, 1.00, 1.00);
    float3 d = float3(0.00, 0.33, 0.67);
    return a + b * cos(6.28318530718 * (c * t + d));
}

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim   = DispatchRaysDimensions().xy;

    float2 ndc = ((float2(pixel) + 0.5) / float2(dim)) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float  aspect   = g_scene.miscParams.x;
    float  tanH     = g_scene.miscParams.y;
    float3 dirView  = normalize(float3(ndc.x * aspect * tanH, ndc.y * tanH, 1.0));
    float3 dirWorld = mul((float3x3)g_scene.viewToWorld, dirView);

    RayDesc r;
    r.Origin    = g_scene.cameraPosition.xyz;
    r.Direction = dirWorld;
    r.TMin      = 0.001;
    r.TMax      = 1000.0;

    Payload p;
    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 0, 0, r, p);

    Output[pixel] = p.color;
}

[shader("miss")]
void Miss(inout Payload p)
{
    float t = saturate(WorldRayDirection().y * 0.5 + 0.5);
    p.color = float4(lerp(float3(0.06, 0.07, 0.10),
                          float3(0.40, 0.55, 0.75), t), 1);
}

[shader("closesthit")]
void Hit(inout Payload p, in Attribs a)
{
    // ClusterID() is the DXR2 system-value intrinsic available in any-hit and
    // closest-hit shaders when the pipeline opts in to ALLOW_CLUSTERED_GEOMETRY.
    // Returns 0xFFFFFFFF when the hit is on a classic (non-cluster) BLAS.
    uint cid = ClusterID();
    float3 base = (cid != 0xFFFFFFFFu) ? ClusterColor(cid)
                                       : float3(0.6, 0.6, 0.6);

    // Use the (normalised) object-space hit position as a stand-in for the
    // surface normal. Exact for spheres centred at the origin (which is most
    // of our test scene) and "good enough decoration" for the torus + cube.
    BuiltInTrianglePositions tri = TriangleObjectPositions();
    float3 baryWeights = float3(1.0 - a.bary.x - a.bary.y, a.bary.x, a.bary.y);
    float3 hitPosObj = tri.p0 * baryWeights.x
                     + tri.p1 * baryWeights.y
                     + tri.p2 * baryWeights.z;
    float3 nObj = normalize(hitPosObj);

    float  ndotl   = saturate(dot(nObj, normalize(float3(0.45, 0.55, -0.7))));
    float  wrap    = ndotl * 0.5 + 0.5;     // half-Lambert
    float  ambient = 0.45;
    float3 lit     = base * (ambient + (1.0 - ambient) * wrap);
    p.color        = float4(lit, 1);
}
