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
    // -------------------------------------------------------------------
    // DATA-DRIVEN material + colour resolution.
    //
    // EVERY per-cluster decision (material override, cluster colour hash
    // key, surface/refraction/reflection tint strength, interior-surface
    // back-face refractivity boost) is read from a SINGLE per-cluster
    // metadata buffer that the CPU populates at scene-build time from
    // each generator's emitted Cluster grid info + the object's scene-
    // config knobs (CheckerConfig, tint multipliers).  The shader has
    // ZERO InstanceID() branches, ZERO cluster-id-range tests, ZERO
    // parity formulas, ZERO oPos-based wall-tile-slot decoding.  Adding
    // a new variant (different checker pattern, new material kind,
    // different tint) is a CPU-only edit.
    // -------------------------------------------------------------------
    const uint        cid  = ClusterID();
    const ClusterMeta meta = LoadClusterMeta(cid);

    MaterialDesc mat = g_materials[InstanceID()];

    // Apply per-cluster material overrides (sentinel <0 = use baseline).
    if (meta.overrideRefl >= 0.0) mat.reflectivity = meta.overrideRefl;
    if (meta.overrideRefr >= 0.0) mat.refractivity = meta.overrideRefr;
    if (meta.overrideIor  >= 0.0) mat.ior          = meta.overrideIor;
    mat.baseColor.xyz *= meta.baseColorScale;

    // Interior surface = part of an enclosed glass volume (slab faces).
    // Back-face hits from inside the glass should appear MORE opaque so
    // the interior surface reads as visible instead of washing out with
    // the transmitted exterior view.
    if ((meta.flags & CLUSTER_META_FLAG_INTERIOR_SURFACE) != 0u &&
        HitKind() == HIT_KIND_TRIANGLE_BACK_FACE &&
        mat.refractivity > 0.0)
    {
        mat.refractivity *= 0.4;
    }

    // Cluster-rainbow palette is gated on a single visualisation knob -
    // miscParams.w in the scene CB.  >0 lets the per-cluster cosine
    // palette tint the base colour (so the user can SEE the cluster
    // boundaries which is the whole point of this sample); 0 makes
    // material colours fully take over.
    const float clusterTint = saturate(g_scene.miscParams.w);
    // Cluster colour lookup uses meta.colorIndex - the CPU sets this to
    // the matching top-tile cid for slab bottom + wall sub-clusters so
    // the entire slab volume column reads as one unit; otherwise it's
    // just the cluster's own cid.
    const float3 clusterCol = (meta.colorIndex != 0xFFFFFFFFu)
                                ? ClusterColor(meta.colorIndex)
                                : float3(0.6, 0.6, 0.6);
    // Surface base = baseColor * lerp(white, clusterCol, clusterTint * meta.surfTintMul).
    // meta.surfTintMul is the per-object knob (default 1.0; floor uses 0.40
    // so its translucent tiles read as pale glass not saturated bricks).
    const float surfTint = clusterTint * meta.surfTintMul;
    const float3 base    = mat.baseColor.xyz * lerp(float3(1, 1, 1), clusterCol, surfTint);

    // World-space surface normal.  Per-vertex normals live in the side-
    // channel keyed by (ClusterID, PrimitiveIndex):
    //   - g_clusterOffsets stores (vertOff, idxOff) per cluster, 8 B/entry
    //   - g_clusterIndices stores 1 uint32 per index (uint8 widened to
    //     uint32 on upload for clean ByteAddressBuffer.Load access)
    //   - g_clusterNormals stores float3 padded to 16 bytes per vertex
    // We compute byte addresses explicitly via ByteAddressBuffer because
    // root-descriptor StructuredBuffer<vector-of-something> has
    // implementation-defined stride padding behaviour - we hit a bug
    // where StructuredBuffer<uint2>'s root-SRV access returned garbage
    // offsets even though the element type is 8 bytes packed on both
    // sides.  Byte loads remove the stride ambiguity.
    uint primIdx     = PrimitiveIndex();
    uint2 off        = g_clusterOffsets.Load2(cid * 8);                   // (vertOff, idxOff)
    uint  idxBase    = (off.y + primIdx * 3) * 4;                         // 4 B per uint32 index
    uint  i0         = g_clusterIndices.Load(idxBase + 0);
    uint  i1         = g_clusterIndices.Load(idxBase + 4);
    uint  i2         = g_clusterIndices.Load(idxBase + 8);
    float3 n0        = asfloat(g_clusterNormals.Load3((off.x + i0) * 16));
    float3 n1        = asfloat(g_clusterNormals.Load3((off.x + i1) * 16));
    float3 n2        = asfloat(g_clusterNormals.Load3((off.x + i2) * 16));
    // a.bary is (w1, w2); w0 = 1 - w1 - w2 (DXR convention).
    float  bw1       = a.bary.x;
    float  bw2       = a.bary.y;
    float  bw0       = 1.0 - bw1 - bw2;
    float3 nObj      = normalize(n0 * bw0 + n1 * bw1 + n2 * bw2);
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
    //   - reflectivity bounces at depths <= 2 (one bounce on the front
    //     face of the surface AND one on the inside back face for glass,
    //     so a glass volume produces TWO visible reflection layers - the
    //     surface highlight and the back-wall internal reflection).
    //     Reflection rays use RAY_FLAG_NONE so they can hit BOTH front
    //     and back faces (essential for internal reflection inside a
    //     glass volume to see the opposite inner wall).
    //   - refractivity bounces at depths <= 4 (enter glass + exit
    //     glass + chain through 2-3 more glass volumes; the inside-the-
    //     glass closesthit at depth 1 hits the BACK face and refracts
    //     OUT, so the camera sees the world behind the glass).  Already
    //     uses RAY_FLAG_NONE so inside-glass rays see the back face.
    //
    // Pipeline MaxRecursionDepth=8 covers the worst-case chain.
    const uint myDepth = p.depth;
    float3 finalColor = surfaceColor;

    // ===================================================================
    // Fresnel-Schlick: reflection probability is angle-dependent.
    //   F0 = reflectance at normal incidence (= mat.reflectivity)
    //   F  = F0 + (1-F0) * (1 - N.V)^5
    // F drives the reflection-mix weight; (1-F)*Tnorm drives the
    // transmission-mix weight (Tnorm normalises so the normal-incidence
    // look matches what we had before the Fresnel switch).
    // ===================================================================
    float NdotV  = saturate(dot(nWorld, -WorldRayDirection()));
    float F0     = mat.reflectivity;
    float F      = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
    float Tnorm  = (mat.refractivity > 0.0 && F0 < 0.999) ? (mat.refractivity / (1.0 - F0)) : 0.0;
    float Tw     = (1.0 - F) * Tnorm;

    bool tirHappened = false;

    if (mat.refractivity > 0.0 && myDepth <= 4)
    {
        // Snell ratio eta = n_outside / n_inside for entering, the
        // inverse for exiting. We assume air (n=1) outside.  HitKind()
        // told us which face we hit; we already flipped nWorld so the
        // normal points TOWARD the incoming ray's origin, which is what
        // HLSL refract() expects.
        float eta  = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE) ? (1.0 / mat.ior) : mat.ior;
        float3 incident   = WorldRayDirection();
        float3 refractDir = refract(incident, nWorld, eta);
        // Tint the refracted RGB by the cluster colour (stained-glass).
        // Strength = clusterTint * meta.refrTintMul (default 0.50; floor
        // dials to 0.25 so translucent tiles read cleanly through).
        float3 refractTint = lerp(float3(1, 1, 1), clusterCol, clusterTint * meta.refrTintMul);

        if (dot(refractDir, refractDir) < 0.001)
        {
            // Total internal reflection: refract() returns 0; fall back
            // to a mirror reflection of the incoming ray (stays inside
            // the glass volume - so childInGlass below stays at the
            // CURRENT-ray value, not the toggled one).  TIR captures
            // 100%% of the light (all reflects, none transmits), so we
            // mix it with weight 1 and SKIP the separate reflection
            // branch below to avoid double-counting the same bounce.
            tirHappened = true;
            refractDir = reflect(incident, nWorld);
            uint childInGlass = p.inGlass;       // TIR keeps us in the same medium
            uint cullFlags    = childInGlass ? RAY_FLAG_NONE
                                              : RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
            float3 tirRGB     = TraceBounce(hitPos + nWorld * 0.001,
                                            refractDir,
                                            cullFlags,
                                            myDepth + 1, childInGlass);
            finalColor = lerp(finalColor, tirRGB * refractTint, 1.0);
        }
        else
        {
            // True refraction with Fresnel-weighted contribution.
            // (1-F) drops at grazing angles -> sides of glass go from
            // mostly-transmissive (normal incidence) to mostly-reflective
            // (grazing) - the classic "Fresnel rim" on a glass ball.
            uint childInGlass = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE) ? 1u : 0u;
            uint cullFlags    = childInGlass ? RAY_FLAG_NONE
                                              : RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
            float3 refractedRGB = TraceBounce(hitPos - nWorld * 0.001,
                                              refractDir,
                                              cullFlags,
                                              myDepth + 1, childInGlass);
            finalColor = lerp(finalColor, refractedRGB * refractTint, Tw);
        }
    }

    if (!tirHappened && mat.reflectivity > 0.0 && myDepth <= 2)
    {
        // Mirror bounce.  Reflection STAYS in the same medium as the
        // current ray (ray hits a surface, bounces back into the same
        // half-space it came from), so childInGlass = p.inGlass.  Cull
        // flags chosen to match: in-air bounces use back-face culling,
        // in-glass bounces (e.g. inside the slab reflecting off the
        // opposite inner wall) need RAY_FLAG_NONE.
        uint childInGlass = p.inGlass;
        uint cullFlags    = childInGlass ? RAY_FLAG_NONE
                                          : RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
        float3 reflectDir   = reflect(WorldRayDirection(), nWorld);
        float3 reflectedRGB = TraceBounce(hitPos + nWorld * 0.001,
                                          reflectDir,
                                          cullFlags,
                                          myDepth + 1, childInGlass);
        // Tint the reflected RGB by the cluster colour.  Strength =
        // clusterTint * meta.reflTintMul.  Defaults:
        //   chrome / glass spheres: meta.reflTintMul ~ 1.08 (effective
        //     ~0.70 at default clusterTint=0.65) - high tint so the
        //     cluster grid reads on near-perfect mirrors.
        //   floor: meta.reflTintMul ~ 0.31 (effective ~0.20) - LOW tint
        //     so the directional sky/horizon variation dominates over
        //     cluster identity on the mirror tiles.
        float3 reflectTint = lerp(float3(1, 1, 1), clusterCol, clusterTint * meta.reflTintMul);
        finalColor = lerp(finalColor, reflectedRGB * reflectTint, F);
    }

    p.color = float4(finalColor, 1);
}

