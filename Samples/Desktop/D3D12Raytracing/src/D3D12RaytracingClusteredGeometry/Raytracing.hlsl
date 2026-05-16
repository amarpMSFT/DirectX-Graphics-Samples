// Raytracing.hlsl
//
// Stage C+D: per-instance materials with four kinds:
//   OPAQUE      diffuse + shadow only
//   REFLECTIVE  diffuse + shadow + 1-bounce reflection
//   REFRACTIVE  diffuse + shadow + 1-bounce refraction (Snell, IOR per material)
//   STOCHASTIC  diffuse + shadow + any-hit probabilistic transparency
//
// Two ray indices:
//   0 = primary / reflection / refraction (ClusterHitGroup: closesthit + anyhit)
//   1 = shadow                            (ShadowHitGroup: empty; FORCE_OPAQUE
//                                          + SKIP_CLOSEST_HIT means no shaders
//                                          run on a hit, only ShadowMiss runs
//                                          on a miss)
//
// Pipeline ALLOW_CLUSTERED_GEOMETRY enables ClusterID() in any-hit and
// closest-hit shaders. CLAS builds use ALLOW_DATA_ACCESS so
// TriangleObjectPositions() can return source positions for normal compute.

#define HLSL
#include "RaytracingHlslCompat.h"

RaytracingAccelerationStructure   Scene       : register(t0);
RWTexture2D<float4>               Output      : register(u0);
ConstantBuffer<SceneConstantBuffer> g_scene   : register(b0);
StructuredBuffer<MaterialDesc>    g_materials : register(t1);

struct [raypayload] Payload
{
    // .rgb = output colour; .a = recursion depth as caller-supplied input
    // (0 = primary ray, 1 = first bounce, ...). Closest-hit reads .a to
    // decide whether to recurse further; both miss and closesthit overwrite
    // .rgba on return so the caller doesn't depend on .a coming back.
    float4 color : write(miss, closesthit, caller) : read(caller, closesthit);
};
struct [raypayload] ShadowPayload
{
    bool inShadow : write(miss, caller) : read(caller);
};
struct Attribs { float2 bary; };

// ============================================================================
// WARP-WORKAROUND  (DELETE WHEN WARP IS FIXED)  - cosine-palette per cluster.
// ============================================================================
float3 ClusterColor(uint cid)
{
    float t = float(cid % 16) * (1.0 / 16.0);
    float3 a = float3(0.55, 0.55, 0.55);
    float3 b = float3(0.45, 0.45, 0.45);
    float3 c = float3(1.00, 1.00, 1.00);
    float3 d = float3(0.00, 0.33, 0.67);
    return a + b * cos(6.28318530718 * (c * t + d));
}

float3 GeometricNormalObj(BuiltInTrianglePositions tri)
{
    return normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
}

// Cheap per-triangle hash for stochastic translucency. Stable across frames so
// the noise pattern doesn't shimmer.
float Hash(uint cid, float2 bary)
{
    uint h = cid * 0x9E3779B9u;
    h ^= asuint(bary.x * 1234.567);
    h ^= asuint(bary.y * 8765.432);
    h ^= (h >> 16);
    return frac(float(h) * 2.3283064e-10);
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

// ----------------------------------------------------------------------------
// Shared helper: fire a shadow ray from world-space hitPos along nWorld toward
// the sun and return the visibility scalar (0 = in shadow, 1 = lit).
// ----------------------------------------------------------------------------
float ShadowVisibility(float3 hitPos, float3 nWorld, float3 toSun)
{
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
        0xff, /*RayContrib*/1, /*MultiplierGeoContrib*/0, /*MissIdx*/1,
        shadow, sp);
    return sp.inShadow ? 0.0 : 1.0;
}

// ----------------------------------------------------------------------------
// Shared helper: trace a recursive ray (reflection or refraction) and return
// the resulting color. Uses the primary hit-group (index 0) so reflective /
// refractive surfaces continue dispatching on materials at depth 1.
//
// childDepth is what the child closesthit will see in its incoming
// payload's .a, so it can decide not to recurse further. Caller computes
// `myDepth + 1` and passes it in.
// ----------------------------------------------------------------------------
float3 TraceBounce(float3 origin, float3 dir, uint cullFlags, uint childDepth)
{
    RayDesc r;
    r.Origin    = origin;
    r.Direction = dir;
    r.TMin      = 0.001;
    r.TMax      = 100.0;
    Payload bp;
    bp.color = float4(0, 0, 0, (float)childDepth);
    TraceRay(Scene, cullFlags, 0xff, /*RayContrib*/0, /*MultiplierGeoContrib*/0,
             /*MissIdx*/0, r, bp);
    return bp.color.rgb;
}

[shader("anyhit")]
void AnyHit(inout Payload p, in Attribs a)
{
    // Stochastic translucency: probabilistic per-triangle reject. Only fires
    // for instances that aren't FORCE_OPAQUE - i.e. REFRACTIVE + STOCHASTIC.
    // For REFRACTIVE we always accept (closest-hit handles refraction). For
    // STOCHASTIC we hash and IgnoreHit a fraction of the time.
    MaterialDesc mat = g_materials[InstanceID()];
    if (mat.kind == MAT_KIND_STOCHASTIC)
    {
        uint cid = ClusterID();
        float r = Hash(cid, a.bary);
        if (r < mat.params.y)               // params.y = translucency
            IgnoreHit();
    }
    // OPAQUE / REFLECTIVE never reach here (FORCE_OPAQUE skips any-hit).
    // REFRACTIVE: fall through, accept the hit.
}

[shader("closesthit")]
void Hit(inout Payload p, in Attribs a)
{
    MaterialDesc mat = g_materials[InstanceID()];

    uint cid = ClusterID();
    float3 cluster = (cid != 0xFFFFFFFFu) ? ClusterColor(cid)
                                          : float3(0.6, 0.6, 0.6);
    float3 base = mat.baseColor.xyz * cluster;

    // World-space geometric normal. ObjectToWorld3x4()'s 3x3 sub-matrix is
    // rotation/scale only (translation column dropped).
    BuiltInTrianglePositions tri = TriangleObjectPositions();
    float3 nObj   = GeometricNormalObj(tri);
    float3 nWorld = normalize(mul((float3x3)ObjectToWorld3x4(), nObj));

    // Flip the normal if we hit a back face (refractive surfaces let rays
    // enter and we may end up reading the wrong side).
    if (HitKind() == HIT_KIND_TRIANGLE_BACK_FACE)
        nWorld = -nWorld;

    float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 toSun  = normalize(g_scene.lightDir.xyz);
    float  NdotL  = saturate(dot(nWorld, toSun));

    float visibility = ShadowVisibility(hitPos, nWorld, toSun);
    float ambient    = g_scene.lightDir.w;
    float lit        = ambient + (1.0 - ambient) * NdotL * visibility;
    float3 surfaceColor = base * lit;

    float3 finalColor = surfaceColor;
    // Recursion-cap via payload alpha. p.color.a was set by the caller to
    // (current depth). REFLECTIVE bounces only at depth 0 (one mirror
    // bounce). REFRACTIVE bounces at depths 0 AND 1: at depth 0 we refract
    // INTO the glass; at depth 1 the ray inside the glass hits the back
    // face and we refract OUT, so the camera sees the world behind the
    // glass (the proper thin-medium approximation). Depth 2 closesthits
    // render diffuse-only - their only TraceRay is the shadow ray, which
    // becomes depth 3 (the leaf, MaxRecursionDepth=3 just barely accepts).
    const uint myDepth = (uint)p.color.a;
    if (myDepth == 0 && mat.kind == MAT_KIND_REFLECTIVE)
    {
        // One-bounce mirror reflection. Reflected ray uses back-face culling
        // (we don't expect to enter solid objects).
        float3 reflectDir   = reflect(WorldRayDirection(), nWorld);
        float3 reflectedRGB = TraceBounce(hitPos + nWorld * 0.001,
                                          reflectDir,
                                          RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                                          myDepth + 1);
        finalColor = lerp(surfaceColor, reflectedRGB, mat.params.x);  // params.x = reflectivity
    }
    else if (myDepth <= 1 && mat.kind == MAT_KIND_REFRACTIVE)
    {
        // Two-interface refraction (proper thin-medium glass).
        //   depth 0 hit: front face from outside  -> refract air -> glass
        //   depth 1 hit: back face from inside    -> refract glass -> air
        // Snell ratio eta = n_outside / n_inside for entering, the inverse
        // for exiting. We assume air (n=1) outside the material and the
        // material's IOR inside. HitKind() told us which face we hit; we
        // already flipped nWorld accordingly so the normal always points
        // TOWARD the incoming ray's origin - which is exactly what HLSL
        // refract() expects for `normal` (the second arg).
        float ior  = mat.params.z;
        float eta  = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE) ? (1.0 / ior) : ior;
        float3 incident    = WorldRayDirection();
        float3 refractDir  = refract(incident, nWorld, eta);
        if (dot(refractDir, refractDir) < 0.001)
        {
            // Total internal reflection: refract() returns 0; fall back to
            // a mirror reflection of the incoming ray.
            refractDir = reflect(incident, nWorld);
        }
        // Refracted ray must NOT cull back-facing - we want to allow
        // entering the volume from the outside AND exiting from inside.
        float3 refractedRGB = TraceBounce(hitPos - nWorld * 0.001,
                                          refractDir,
                                          RAY_FLAG_NONE,
                                          myDepth + 1);
        finalColor = lerp(surfaceColor, refractedRGB, mat.params.y);  // params.y = translucency
    }
    // OPAQUE + STOCHASTIC: surfaceColor as-is. STOCHASTIC's "transparency" is
    // delivered by the any-hit shader rejecting some hits on the way in,
    // which means closest-hit never even runs for the rejected triangles.

    p.color = float4(finalColor, 1);
}
