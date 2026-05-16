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
// look them up by triangle corner) ride along in these BYTE-ADDRESS
// buffers, indexed by ClusterID() + PrimitiveIndex() in the closesthit.
//
// We use ByteAddressBuffer (not StructuredBuffer<T>) because root SRVs +
// vector-typed StructuredBuffers (uint2, float3) have implementation-
// defined stride padding behaviour - we hit a bug where
// `StructuredBuffer<uint2>` returned garbage offsets even though the
// element type was 8 bytes packed.  ByteAddressBuffer + manual
// offsets eliminates the ambiguity:  WE compute the byte address, the
// runtime just does a raw load.
ByteAddressBuffer                 g_clusterNormals : register(t2);   // float3 stored as 16 bytes (with .w padding)
ByteAddressBuffer                 g_clusterIndices : register(t3);   // uint  stored as  4 bytes
ByteAddressBuffer                 g_clusterOffsets : register(t4);   // uint2 stored as  8 bytes (vertOff, idxOff per cluster)
// Per-cluster GENERIC metadata buffer (ClusterMeta, 48B per cluster).
// Drives ALL per-cluster material / colour decisions in the closesthit -
// no in-shader InstanceID() branches, no cid-range hardcoding, no parity
// formulas, no wall-tile-slot oPos decoding.  Every per-cluster decision
// lives in CPU-authored data; the shader just loads the meta and applies
// the override fields uniformly.  See RaytracingHlslCompat.h ClusterMeta.
ByteAddressBuffer                 g_clusterMeta    : register(t5);

ClusterMeta LoadClusterMeta(uint cid)
{
    // Stride = 48 bytes (matches CPU-side sizeof(ClusterMeta)).  Plain
    // 4-byte loads packed into the struct, asfloat() for the float fields.
    const uint base = cid * 48u;
    ClusterMeta m;
    m.colorIndex     = g_clusterMeta.Load(base +  0);
    m.flags          = g_clusterMeta.Load(base +  4);
    m.overrideRefl   = asfloat(g_clusterMeta.Load(base +  8));
    m.overrideRefr   = asfloat(g_clusterMeta.Load(base + 12));
    m.overrideIor    = asfloat(g_clusterMeta.Load(base + 16));
    m.baseColorScale = asfloat(g_clusterMeta.Load(base + 20));
    m.surfTintMul    = asfloat(g_clusterMeta.Load(base + 24));
    m.refrTintMul    = asfloat(g_clusterMeta.Load(base + 28));
    m.reflTintMul    = asfloat(g_clusterMeta.Load(base + 32));
    m._pad0          = 0;
    m._pad1          = 0;
    m._pad2          = 0;
    return m;
}

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
    // 1 = ray is currently INSIDE a glass volume (entered via a refraction
    // through a glass front face).  closesthit reads this to decide whether
    // back-face culling is OK on the next bounce - inside-glass child rays
    // need RAY_FLAG_NONE so they can hit the BACK face of the glass to
    // refract out, while air-side rays use RAY_FLAG_CULL_BACK_FACING_-
    // TRIANGLES so they don't see "inside" opaque or grazing-angle objects.
    uint   inGlass : write(caller)      : read(closesthit);
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
    // Murmurhash3 finalizer.  Mixes ALL bits of cid into the output so
    // neighbours in EITHER direction of the parametric grid (cid+1 in
    // the longitude direction, cid+tilesLong in the latitude direction)
    // get maximally-different hue indices.  Plain `cid % 16` produced
    // smooth palette stripes; this gives a true CHECKER scatter on
    // every mesh that lays clusters out in row-major order.
    uint h = cid;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    h &= 0x3fu;                                  // 6 bits -> 64 hues
    float t = float(h) * (1.0 / 64.0);
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
        p.color   = float4(0, 0, 0, 1);
        p.depth   = 0;
        p.inGlass = 0;       // primary rays start in air
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

    // Daytime sky.  Sun is up (lightDir.y ~ 0.75 = ~50° elevation), so the
    // dome reads as DAYLIGHT BLUE - no sunset orange anywhere.  Three
    // bands: deep sky-blue overhead -> mid sky -> hazy lighter blue at
    // the horizon (atmospheric scatter).  The sun itself is a tight
    // disc rendered in the sun-glow code below.
    float3 zenith   = float3(0.18, 0.42, 0.82);  // deep sky blue overhead
    float3 midSky   = float3(0.42, 0.62, 0.90);  // mid sky
    float3 horizon  = float3(0.68, 0.80, 0.92);  // hazy blue at horizon
    float  tSky     = saturate(y);               // 0 at horizon, 1 at zenith
    float3 sky      = lerp(horizon,
                           lerp(midSky, zenith, smoothstep(0.0, 0.55, tSky)),
                           smoothstep(0.0, 0.25, tSky));
                                                           // band into the lower
                                                           // 20% so the rest of
                                                           // the dome reads
                                                           // BLUE not orange.

    // Ground band below the horizon - moderately darker at the horizon
    // for atmospheric perspective, full sand colour up close.  Soft
    // falloff so the dark band reads as haze, not as a black void.
    // Desaturated palette - reads as warm beige/khaki, not punchy
    // orange-brown (the previous gradient was too saturated and pulled
    // the eye away from the colourful clustered objects above it).
    float3 ground   = lerp(float3(0.50, 0.47, 0.42),    // moderately dark, low-saturation beige at horizon (FAR)
                           float3(0.86, 0.82, 0.74),    // light warm beige under foot                       (NEAR)
                           saturate(-y * 4.0));

    // Sun disc/halo.  Only fires when the ray direction is close to the
    // actual sun direction.  Daytime sun is white-warm, not sunset-orange.
    float3 sunDir   = normalize(g_scene.lightDir.xyz);
    float  sunAlign = saturate(dot(d, sunDir));
    float3 sunGlow  = float3(1.10, 1.00, 0.85) *
                      pow(sunAlign, 24.0) *                 // tight sun disc
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
float3 TraceBounce(float3 origin, float3 dir, uint cullFlags, uint childDepth, uint childInGlass)
{
    RayDesc r;
    r.Origin    = origin;
    r.Direction = dir;
    r.TMin      = 0.001;
    r.TMax      = 100.0;
    Payload bp;
    bp.color   = float4(0, 0, 0, 1);
    bp.depth   = childDepth;        // dedicated write(caller)/read(closesthit) field
    bp.inGlass = childInGlass;      // 1 if the new ray is travelling INSIDE a glass volume
    TraceRay(Scene, cullFlags, 0xff, /*RayContrib*/0, /*MultiplierGeoContrib*/0,
             /*MissIdx*/0, r, bp);
    return bp.color.rgb;
}

[shader("anyhit")]
void GlassAnyHit(inout Payload p, in Attribs a)
{
    // Stochastic translucency for the GLASS hit group only.
    // OpaqueHitGroup has NO any-hit shader bound - opaque-instance
    // traversals skip the any-hit dispatch entirely (cheaper).
    //   (translucency>0)  : maybe IgnoreHit
    //   (refractivity>0)  : always accept (closesthit handles refraction)
    //   (both>0)          : maybe IgnoreHit; if accepted, closesthit refracts
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

// ============================================================================
// Shared closesthit context.  Loaded once at the top of either OpaqueHit
// or GlassHit so the shared work (ClusterMeta load, per-cluster material
// overrides, vertex-normal smoothing, shadow tracing, surface base
// colour computation) lives in ONE place even though we ship TWO
// specialised closesthit shaders.
// ============================================================================
struct HitContext
{
    ClusterMeta  meta;
    MaterialDesc mat;
    float3       nWorld;        // world-space surface normal (back-face-flipped if allowBackFaceFlip)
    float3       hitPos;        // world-space hit position
    float3       base;          // baseColor * cluster tint * lit visibility (final surface colour)
    float3       clusterCol;
    float        clusterTint;   // global slider (g_scene.miscParams.w)
};

void LoadHitContext(in Attribs a, in bool allowBackFaceFlip, out HitContext ctx)
{
    const uint cid = ClusterID();
    ctx.meta = LoadClusterMeta(cid);
    ctx.mat  = g_materials[InstanceID()];

    // Per-cluster material overrides (sentinel <0 = no override).
    if (ctx.meta.overrideRefl >= 0.0) ctx.mat.reflectivity = ctx.meta.overrideRefl;
    if (ctx.meta.overrideRefr >= 0.0) ctx.mat.refractivity = ctx.meta.overrideRefr;
    if (ctx.meta.overrideIor  >= 0.0) ctx.mat.ior          = ctx.meta.overrideIor;
    ctx.mat.baseColor.xyz *= ctx.meta.baseColorScale;

    // Interior-surface back-face refractivity boost (slab faces only;
    // shader skips the test cheaply when the flag bit is 0).
    if ((ctx.meta.flags & CLUSTER_META_FLAG_INTERIOR_SURFACE) != 0u &&
        HitKind() == HIT_KIND_TRIANGLE_BACK_FACE &&
        ctx.mat.refractivity > 0.0)
    {
        ctx.mat.refractivity *= 0.4;
    }

    // Cluster colour + surface base.
    ctx.clusterTint = saturate(g_scene.miscParams.w);
    ctx.clusterCol  = (ctx.meta.colorIndex != 0xFFFFFFFFu)
                        ? ClusterColor(ctx.meta.colorIndex)
                        : float3(0.6, 0.6, 0.6);
    const float  surfTint  = ctx.clusterTint * ctx.meta.surfTintMul;
    const float3 baseUnlit = ctx.mat.baseColor.xyz
                            * lerp(float3(1, 1, 1), ctx.clusterCol, surfTint);

    // World-space surface normal: smoothed via per-vertex cluster-normal
    // side-channel keyed by (ClusterID, PrimitiveIndex).
    uint primIdx = PrimitiveIndex();
    uint2 off    = g_clusterOffsets.Load2(cid * 8);
    uint  idxBase= (off.y + primIdx * 3) * 4;
    uint  i0     = g_clusterIndices.Load(idxBase + 0);
    uint  i1     = g_clusterIndices.Load(idxBase + 4);
    uint  i2     = g_clusterIndices.Load(idxBase + 8);
    float3 n0    = asfloat(g_clusterNormals.Load3((off.x + i0) * 16));
    float3 n1    = asfloat(g_clusterNormals.Load3((off.x + i1) * 16));
    float3 n2    = asfloat(g_clusterNormals.Load3((off.x + i2) * 16));
    float  bw1   = a.bary.x;
    float  bw2   = a.bary.y;
    float  bw0   = 1.0 - bw1 - bw2;
    float3 nObj  = normalize(n0 * bw0 + n1 * bw1 + n2 * bw2);
    ctx.nWorld   = normalize(mul((float3x3)ObjectToWorld3x4(), nObj));

    // Glass surfaces can be hit on the back face (refraction inside the
    // volume); flip the normal so it points TOWARD the incoming ray and
    // HLSL refract() sees the geometry HLSL expects.  Opaque surfaces
    // shouldn't see back faces (FORCE_OPAQUE + back-face culling on
    // reflection rays), so the caller passes allowBackFaceFlip=false.
    if (allowBackFaceFlip && HitKind() == HIT_KIND_TRIANGLE_BACK_FACE)
        ctx.nWorld = -ctx.nWorld;

    ctx.hitPos       = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 toSun     = normalize(g_scene.lightDir.xyz);
    float  NdotL     = saturate(dot(ctx.nWorld, toSun));
    float  visibility= ShadowVisibility(ctx.hitPos, ctx.nWorld, toSun);
    float  ambient   = g_scene.lightDir.w;
    float  lit       = ambient + (1.0 - ambient) * NdotL * visibility;
    ctx.base         = baseUnlit * lit;
}

// ============================================================================
// OpaqueHit - specialised closesthit for instances whose materials are
// FORCE_OPAQUE everywhere (chrome, polished metal, pure matte).
//
// No refraction logic.  No Fresnel composition.  No back-face handling.
// Just lit-surface + optional single mirror reflection bounce.  Paired
// with OpaqueHitGroup which has NO any-hit shader bound - so the
// any-hit dispatch is skipped entirely for opaque traversals (the
// "make-it-fast-for-the-chrome-ball" path the user asked for).
//
// Per-cluster material variation via ClusterMeta still works as long
// as the overrides keep refractivity at 0 (mirror or matte).
// ============================================================================
[shader("closesthit")]
void OpaqueHit(inout Payload p, in Attribs a)
{
    HitContext ctx;
    LoadHitContext(a, /*allowBackFaceFlip*/false, ctx);

    float3 finalColor = ctx.base;

    if (ctx.mat.reflectivity > 0.0 && p.depth <= 2)
    {
        // Mirror bounce.  Opaque-only path: reflection ray uses
        // RAY_FLAG_CULL_BACK_FACING_TRIANGLES (we're in air, never
        // inside a glass volume on an opaque-instance reflection).
        float3 reflectDir   = reflect(WorldRayDirection(), ctx.nWorld);
        float3 reflectedRGB = TraceBounce(ctx.hitPos + ctx.nWorld * 0.001,
                                          reflectDir,
                                          RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                                          p.depth + 1, /*inGlass*/0u);
        float3 reflectTint  = lerp(float3(1, 1, 1), ctx.clusterCol,
                                   ctx.clusterTint * ctx.meta.reflTintMul);
        finalColor = lerp(finalColor, reflectedRGB * reflectTint, ctx.mat.reflectivity);
    }

    p.color = float4(finalColor, 1);
}

// ============================================================================
// GlassHit - full Fresnel-Schlick closesthit for any instance with
// refraction in its material chain (baseline OR per-cluster checker
// override).  Paired with GlassHitGroup which has GlassAnyHit bound.
// ============================================================================
[shader("closesthit")]
void GlassHit(inout Payload p, in Attribs a)
{
    HitContext ctx;
    LoadHitContext(a, /*allowBackFaceFlip*/true, ctx);

    const uint myDepth = p.depth;
    float3 finalColor  = ctx.base;

    // Fresnel-Schlick weighting (angle-dependent reflection probability).
    float NdotV = saturate(dot(ctx.nWorld, -WorldRayDirection()));
    float F0    = ctx.mat.reflectivity;
    float F     = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
    float Tnorm = (ctx.mat.refractivity > 0.0 && F0 < 0.999)
                    ? (ctx.mat.refractivity / (1.0 - F0)) : 0.0;
    float Tw    = (1.0 - F) * Tnorm;

    bool tirHappened = false;

    if (ctx.mat.refractivity > 0.0 && myDepth <= 4)
    {
        float  eta        = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE)
                                ? (1.0 / ctx.mat.ior) : ctx.mat.ior;
        float3 incident   = WorldRayDirection();
        float3 refractDir = refract(incident, ctx.nWorld, eta);
        float3 refractTint= lerp(float3(1, 1, 1), ctx.clusterCol,
                                 ctx.clusterTint * ctx.meta.refrTintMul);

        if (dot(refractDir, refractDir) < 0.001)
        {
            // Total internal reflection: full mirror, stays in same medium.
            tirHappened = true;
            refractDir = reflect(incident, ctx.nWorld);
            uint childInGlass = p.inGlass;
            uint cullFlags    = childInGlass ? RAY_FLAG_NONE
                                              : RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
            float3 tirRGB     = TraceBounce(ctx.hitPos + ctx.nWorld * 0.001,
                                            refractDir, cullFlags,
                                            myDepth + 1, childInGlass);
            finalColor = lerp(finalColor, tirRGB * refractTint, 1.0);
        }
        else
        {
            // True refraction with Fresnel-weighted transmission.
            uint childInGlass = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE) ? 1u : 0u;
            uint cullFlags    = childInGlass ? RAY_FLAG_NONE
                                              : RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
            float3 refractedRGB = TraceBounce(ctx.hitPos - ctx.nWorld * 0.001,
                                              refractDir, cullFlags,
                                              myDepth + 1, childInGlass);
            finalColor = lerp(finalColor, refractedRGB * refractTint, Tw);
        }
    }

    if (!tirHappened && ctx.mat.reflectivity > 0.0 && myDepth <= 2)
    {
        uint childInGlass = p.inGlass;
        uint cullFlags    = childInGlass ? RAY_FLAG_NONE
                                          : RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
        float3 reflectDir   = reflect(WorldRayDirection(), ctx.nWorld);
        float3 reflectedRGB = TraceBounce(ctx.hitPos + ctx.nWorld * 0.001,
                                          reflectDir, cullFlags,
                                          myDepth + 1, childInGlass);
        float3 reflectTint  = lerp(float3(1, 1, 1), ctx.clusterCol,
                                   ctx.clusterTint * ctx.meta.reflTintMul);
        finalColor = lerp(finalColor, reflectedRGB * reflectTint, F);
    }

    p.color = float4(finalColor, 1);
}

