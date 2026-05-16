// Raytracing.hlsl
//
// Stage B: directional sun + shadow rays + ambient-floor lighting.
//
// Pipeline is created with D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_CLUSTERED_GEOMETRY
// so ClusterID() returns the actual cluster id on hits into a Cluster BLAS
// (and CLUSTER_ID_INVALID = 0xFFFFFFFF on hits into a classic triangle BLAS).
//
// Two ray indices:
//   0 = primary ray  -> Miss, ClusterHitGroup (closest-hit)
//   1 = shadow ray   -> ShadowMiss, ShadowHitGroup (no shaders run; we use
//                       FORCE_OPAQUE | SKIP_CLOSEST_HIT | ACCEPT_FIRST_HIT
//                       so the only thing we want from a shadow ray is "did
//                       it miss" -> not in shadow).

#define HLSL
#include "RaytracingHlslCompat.h"

RaytracingAccelerationStructure Scene  : register(t0);
RWTexture2D<float4>             Output : register(u0);
ConstantBuffer<SceneConstantBuffer> g_scene : register(b0);

struct [raypayload] Payload
{
    float4 color : write(miss, closesthit) : read(caller);
};
struct [raypayload] ShadowPayload
{
    // Pre-init to "occluded"; ShadowMiss flips to false. With
    // SKIP_CLOSEST_HIT_SHADER + FORCE_OPAQUE the only shader that can run
    // on a shadow ray is the miss shader, so this is a 1-bit answer.
    bool inShadow : write(miss, caller) : read(caller);
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
// Workaround: cosine-palette (Inigo Quilez style). Same 16-step rainbow
// visual outcome, no static-const array-of-vec lookup. Once WARP picks up
// the fix, this function can be replaced with the natural array indexing.
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

// Geometric normal of the hit triangle in object space.
// `normalize(hitPosObj)` only happens to work for spheres centred at the
// origin; everything else (the floor especially) needs a true geo normal,
// computed from the triangle's actual edges via TriangleObjectPositions().
// Spec note: this intrinsic requires the CLAS build to have been issued
// with D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS - see the spec-audit
// comment in BuildSharedClusterTrianglesInputs.
float3 GeometricNormalObj(BuiltInTrianglePositions tri)
{
    return normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
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
    p.color = float4(0, 0, 0, 1);
    TraceRay(Scene,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        /*InstanceInclusionMask*/0xff,
        /*RayContributionToHitGroupIndex*/0,
        /*MultiplierForGeometryContributionToHitGroupIndex*/0,
        /*MissShaderIndex*/0,
        r, p);

    Output[pixel] = p.color;
}

[shader("miss")]
void Miss(inout Payload p)
{
    float t = saturate(WorldRayDirection().y * 0.5 + 0.5);
    p.color = float4(lerp(float3(0.06, 0.07, 0.10),
                          float3(0.40, 0.55, 0.75), t), 1);
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload p)
{
    p.inShadow = false;
}

[shader("closesthit")]
void Hit(inout Payload p, in Attribs a)
{
    // ClusterID() is the DXR2 system-value intrinsic available in any-hit
    // and closest-hit shaders when the pipeline opts in to
    // ALLOW_CLUSTERED_GEOMETRY. Returns 0xFFFFFFFF when the hit is on a
    // classic (non-cluster) BLAS - we don't have any of those in this
    // sample but the shader is defensive about it anyway.
    uint cid = ClusterID();
    float3 base = (cid != 0xFFFFFFFFu) ? ClusterColor(cid)
                                       : float3(0.6, 0.6, 0.6);

    // World-space geometric normal: cross product of triangle edges in
    // object space, then transform by ObjectToWorld3x4()'s rotation/scale
    // (cast to 3x3 so the translation column is dropped).
    BuiltInTrianglePositions tri = TriangleObjectPositions();
    float3 nObj   = GeometricNormalObj(tri);
    float3 nWorld = normalize(mul((float3x3)ObjectToWorld3x4(), nObj));

    float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 toSun  = normalize(g_scene.lightDir.xyz);
    float  NdotL  = saturate(dot(nWorld, toSun));

    // Shadow ray. Origin offset along the surface normal by epsilon to
    // avoid self-intersect. FORCE_OPAQUE | SKIP_CLOSEST_HIT |
    // ACCEPT_FIRST_HIT means the only shader that can run on this trace is
    // ShadowMiss - it sets inShadow=false. Any geometry hit is treated as
    // opaque and short-circuits the ray immediately. The pre-init
    // sp.inShadow=true is what survives a hit.
    RayDesc shadow;
    shadow.Origin    = hitPos + nWorld * 0.001;
    shadow.Direction = toSun;
    shadow.TMin      = 0.001;
    shadow.TMax      = 100.0;
    ShadowPayload sp;
    sp.inShadow = true;
    TraceRay(Scene,
        RAY_FLAG_FORCE_OPAQUE
            | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
            | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
            | RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        /*InstanceInclusionMask*/0xff,
        /*RayContributionToHitGroupIndex*/1,
        /*MultiplierForGeometryContributionToHitGroupIndex*/0,
        /*MissShaderIndex*/1,
        shadow, sp);

    float visibility = sp.inShadow ? 0.0 : 1.0;
    float ambient    = g_scene.lightDir.w;            // [0..1]
    float lit        = ambient + (1.0 - ambient) * NdotL * visibility;
    p.color          = float4(base * lit, 1);
}
