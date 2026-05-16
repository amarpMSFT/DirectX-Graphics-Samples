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

// Per-cluster shader-side buffers carrying smooth normals + the cluster
// index buffer.  DXR2 cluster geometry's CLAS only consumes positions in
// its vertex buffer, so per-vertex normals (and the indices we need to
// look them up by triangle corner) ride along in these structured
// buffers, indexed by ClusterID() + PrimitiveIndex() in the closesthit.
//
// Normals are stored as float4 (with .w = 0 padding) instead of float3 so
// the HLSL StructuredBuffer<> stride matches the C++ upload exactly -
// `StructuredBuffer<float3>` with a root SRV has implementation-defined
// stride padding behaviour that has bitten us before.  `float4` is an
// unambiguous 16-byte stride on both sides.
StructuredBuffer<float4>          g_clusterNormals : register(t2);
StructuredBuffer<uint>            g_clusterIndices : register(t3);
StructuredBuffer<uint2>           g_clusterOffsets : register(t4);

struct [raypayload] Payload
{
    float4 color : write(miss, closesthit) : read(caller);
    // Explicit recursion-depth field. caller writes (raygen=0; bounced
    // closesthit caller writes parentDepth+1), closesthit reads to decide
    // whether to keep recursing. Kept separate from .color (rather than
    // packed into .color.a) because the SM 6.10 raypayload qualifier's
    // mixed write(caller, miss, closesthit) on the same field doesn't
    // reliably propagate caller writes through to the closesthit on this
    // driver; using a dedicated field with single-direction write(caller)
    // + read(closesthit) qualifiers sidesteps the issue.
    uint   depth : write(caller)        : read(closesthit);
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

    float aspect = g_scene.miscParams.x;
    float tanH   = g_scene.miscParams.y;
    // Sub-pixel sample count comes from the host (1, 2, or 4) via the
    // SceneConstantBuffer.  Floats are easier to ship across the CB than
    // ints; we round + clamp to the nearest valid count.
    uint  samples = (uint)(g_scene.miscParams.z + 0.5);
    if (samples >= 4)      samples = 4;
    else if (samples >= 2) samples = 2;
    else                   samples = 1;

    // Sub-pixel sample positions in [0,1]^2 within the pixel.  4-sample =
    // standard 4-rotated-grid 4xMSAA pattern (hits 4 distinct horizontal
    // AND vertical positions, breaks staircase on both axes).  2-sample =
    // diagonal pair.  1-sample = pixel centre.  Each sample arrives at the
    // same closesthit at slightly different barycentrics, so the Hash() in
    // the stochastic any-hit also returns N different random values per
    // pixel - 4x supersampling does free noise reduction on the frosted-
    // glass surfaces in addition to edge antialiasing.
    static const float2 kSubPixel4[4] = {
        float2(0.125, 0.625),
        float2(0.375, 0.125),
        float2(0.625, 0.875),
        float2(0.875, 0.375),
    };
    static const float2 kSubPixel2[2] = {
        float2(0.25, 0.25),
        float2(0.75, 0.75),
    };

    float3 accum = float3(0, 0, 0);
    for (uint s = 0; s < samples; ++s)
    {
        float2 sub;
        if      (samples == 4) sub = kSubPixel4[s];
        else if (samples == 2) sub = kSubPixel2[s];
        else                   sub = float2(0.5, 0.5);

        float2 ndc = ((float2(pixel) + sub) / float2(dim)) * 2.0 - 1.0;
        ndc.y = -ndc.y;

        float3 dirView  = normalize(float3(ndc.x * aspect * tanH, ndc.y * tanH, 1.0));
        float3 dirWorld = mul((float3x3)g_scene.viewToWorld, dirView);

        RayDesc r;
        r.Origin    = g_scene.cameraPosition.xyz;
        r.Direction = dirWorld;
        r.TMin      = 0.001;
        r.TMax      = 1000.0;

        Payload p;
        p.color = float4(0, 0, 0, 1);
        p.depth = 0;
        TraceRay(Scene,
            RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            /*InstanceInclusionMask*/0xff,
            /*RayContributionToHitGroupIndex*/0,
            /*MultiplierForGeometryContributionToHitGroupIndex*/0,
            /*MissShaderIndex*/0,
            r, p);
        accum += p.color.rgb;
    }

    Output[pixel] = float4(accum / (float)samples, 1);
}

[shader("miss")]
void Miss(inout Payload p)
{
    // Sunset sky / ground gradient.  The ray direction tells us where the
    // camera is looking - we synthesise a believable environment from the
    // y component:
    //
    //   y > 0       sky.   Top is deep dusk-blue, fading through orange
    //               near the horizon as if looking toward the setting sun.
    //   y < 0       ground.  Mostly desaturated dirt-brown.
    //   y around 0  horizon band where the sun is - we add a warm rim
    //               highlight that's hottest exactly at y=0.
    //
    // No texture, no skybox - cheap procedural that ties the lit colours
    // (which use the same sun direction in g_scene.lightDir) to a
    // matching environment so reflections and refractions of the sky
    // carry the right colour temperature.
    float3 d = normalize(WorldRayDirection());
    float  y = d.y;

    // Two sky bands above the horizon.
    float3 zenith   = float3(0.10, 0.13, 0.30);  // deep dusk
    float3 midSky   = float3(0.45, 0.32, 0.40);  // mauve transition
    float3 horizon  = float3(0.95, 0.55, 0.25);  // warm sunset orange
    float  tSky     = saturate(y);               // 0 at horizon, 1 at zenith
    float3 sky      = lerp(horizon,
                           lerp(midSky, zenith, smoothstep(0.0, 0.6, tSky)),
                           smoothstep(0.0, 0.4, tSky));

    // Ground band below the horizon - dirt brown deepening with depth.
    float3 ground   = lerp(float3(0.42, 0.30, 0.18),
                           float3(0.10, 0.07, 0.05),
                           saturate(-y * 1.4));

    // Sun glow at the horizon facing the actual sun.  When the ray
    // direction is roughly aligned with the sun's azimuth the warm tone
    // brightens further, simulating the corona / atmospheric scatter.
    float3 sunDir   = normalize(g_scene.lightDir.xyz);
    float  sunAlign = saturate(dot(d, sunDir));
    float3 sunGlow  = float3(1.20, 0.70, 0.30) *
                      pow(sunAlign, 8.0) *
                      smoothstep(-0.05, 0.30, y);  // only above horizon

    // Composite.
    float3 col = (y >= 0.0) ? sky : ground;
    col += sunGlow;
    p.color = float4(col, 1);
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
    bp.color = float4(0, 0, 0, 1);
    bp.depth = childDepth;          // dedicated write(caller)/read(closesthit) field
    TraceRay(Scene, cullFlags, 0xff, /*RayContrib*/0, /*MultiplierGeoContrib*/0,
             /*MissIdx*/0, r, bp);
    return bp.color.rgb;
}

[shader("anyhit")]
void AnyHit(inout Payload p, in Attribs a)
{
    // Stochastic translucency: probabilistic per-triangle reject. Only
    // fires for instances that aren't FORCE_OPAQUE - which is set when
    // BOTH translucency==0 AND refractivity==0.  So we get here for
    // (translucency>0)  : maybe IgnoreHit
    // (refractivity>0)  : always accept (closesthit handles refraction)
    // (both>0)          : maybe IgnoreHit; if accepted, closesthit refracts
    MaterialDesc mat = g_materials[InstanceID()];
    if (mat.translucency > 0.0)
    {
        uint cid = ClusterID();
        float r = Hash(cid, a.bary);
        if (r < mat.translucency)
            IgnoreHit();
    }
    // Otherwise fall through, accept the hit.
}

[shader("closesthit")]
void Hit(inout Payload p, in Attribs a)
{
    MaterialDesc mat = g_materials[InstanceID()];

    uint cid = ClusterID();
    // Cluster-rainbow palette is gated on a single visualisation knob -
    // miscParams.w in the scene CB.  >0 lets the per-cluster cosine
    // palette tint the base colour (so the user can SEE the cluster
    // boundaries which is the whole point of this sample); 0 makes
    // material colours fully take over.  Default is a subtle 0.3 blend.
    float  clusterTint = saturate(g_scene.miscParams.w);
    float3 clusterCol  = (cid != 0xFFFFFFFFu) ? ClusterColor(cid)
                                              : float3(0.6, 0.6, 0.6);
    float3 base = mat.baseColor.xyz * lerp(float3(1, 1, 1), clusterCol, clusterTint);

    // World-space surface normal.  Per-vertex normals live in the side-
    // channel (g_clusterNormals + g_clusterIndices indexed by ClusterID()
    // and PrimitiveIndex()); we fetch the three corner normals and
    // barycentric-interpolate.  Smooth shading on curved surfaces (sphere
    // / torus / Klein) was the user's #5 ask; the cube + floor have
    // constant per-cluster normals so the interpolation degenerates to
    // flat there for free.
    uint primIdx = PrimitiveIndex();
    uint2 off    = g_clusterOffsets[cid];
    uint i0      = g_clusterIndices[off.y + primIdx * 3 + 0];
    uint i1      = g_clusterIndices[off.y + primIdx * 3 + 1];
    uint i2      = g_clusterIndices[off.y + primIdx * 3 + 2];
    float3 n0    = g_clusterNormals[off.x + i0];
    float3 n1    = g_clusterNormals[off.x + i1];
    float3 n2    = g_clusterNormals[off.x + i2];
    // a.bary is (w1, w2); w0 = 1 - w1 - w2 (DXR convention).
    float  bw1   = a.bary.x;
    float  bw2   = a.bary.y;
    float  bw0   = 1.0 - bw1 - bw2;
    float3 nObj  = normalize(n0 * bw0 + n1 * bw1 + n2 * bw2);
    // ObjectToWorld3x4()'s 3x3 sub-matrix is rotation/scale only.  All
    // our instances are uniform-scale so direct mul is correct here; for
    // non-uniform scale we'd need the inverse-transpose of the upper-3x3.
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

    // Compose optical effects.  Order: surface -> refraction -> reflection.
    // Each effect is a `lerp` so it composites over whatever's already in
    // finalColor at that point.  Recursion-cap via the dedicated payload
    // .depth field (caller writes, closesthit reads):
    //
    //   - reflectivity bounces only at depth 0 (one mirror bounce)
    //   - refractivity bounces at depths 0 AND 1 (enter glass + exit
    //     glass = the proper thin-medium approximation; the inside-the-
    //     glass closesthit at depth 1 hits the BACK face and refracts
    //     OUT, so the camera sees the world behind the glass).
    //
    // Pipeline MaxRecursionDepth=3 means the depth-2 refraction-exit's
    // shadow ray (which becomes depth 3) is the leaf.
    const uint myDepth = p.depth;
    float3 finalColor = surfaceColor;

    if (mat.refractivity > 0.0 && myDepth <= 1)
    {
        // Snell ratio eta = n_outside / n_inside for entering, the
        // inverse for exiting. We assume air (n=1) outside.  HitKind()
        // told us which face we hit; we already flipped nWorld so the
        // normal points TOWARD the incoming ray's origin, which is what
        // HLSL refract() expects.
        float eta  = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE) ? (1.0 / mat.ior) : mat.ior;
        float3 incident   = WorldRayDirection();
        float3 refractDir = refract(incident, nWorld, eta);
        if (dot(refractDir, refractDir) < 0.001)
        {
            // Total internal reflection: refract() returns 0; fall back
            // to a mirror reflection of the incoming ray.
            refractDir = reflect(incident, nWorld);
        }
        // Refracted ray must NOT cull back-facing - we want to allow
        // entering the volume from the outside AND exiting from inside.
        float3 refractedRGB = TraceBounce(hitPos - nWorld * 0.001,
                                          refractDir,
                                          RAY_FLAG_NONE,
                                          myDepth + 1);
        finalColor = lerp(finalColor, refractedRGB, mat.refractivity);
    }

    if (mat.reflectivity > 0.0 && myDepth == 0)
    {
        // One-bounce mirror reflection.  Reflected ray uses back-face
        // culling - we don't expect to enter solid objects from a mirror
        // bounce.
        float3 reflectDir   = reflect(WorldRayDirection(), nWorld);
        float3 reflectedRGB = TraceBounce(hitPos + nWorld * 0.001,
                                          reflectDir,
                                          RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                                          myDepth + 1);
        finalColor = lerp(finalColor, reflectedRGB, mat.reflectivity);
    }

    p.color = float4(finalColor, 1);
}


