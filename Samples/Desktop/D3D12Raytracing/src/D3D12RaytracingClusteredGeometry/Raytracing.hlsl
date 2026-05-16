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
    MaterialDesc mat = g_materials[InstanceID()];

    uint cid = ClusterID();

    // ---- Per-cluster material override demo --------------------------
    // Sphere2 (instanceID=2, the matte aqua opaque sphere) gets a CHECKER
    // pattern: alternate clusters are "matte aqua" (default material) vs
    // "translucent aqua glass" (refractive override).  Demonstrates that
    // material can vary PER CLUSTER inside a single BLAS - a useful
    // pattern for things like decals, damage, or mosaic tiles, since
    // each cluster has its own ClusterID() and the closesthit can use
    // it to look up cluster-specific data.
    //
    // Sphere2 mesh: numLat=16 numLong=32 tileLat=4 tileLong=4 ->
    // tilesLat=4, tilesLong=8 -> 32 clusters with IDs 200..231.
    // Cluster (latIdx, longIdx) has localID = latIdx*8 + longIdx.
    // Checker = (latIdx + longIdx) parity.
    if (InstanceID() == 2)
    {
        const uint kSphere2FirstCid = 200;
        const uint kSphere2TilesLong = 8;
        uint local   = cid - kSphere2FirstCid;
        uint latIdx  = local / kSphere2TilesLong;
        uint longIdx = local % kSphere2TilesLong;
        bool isB     = ((latIdx + longIdx) & 1u) != 0u;
        if (isB)
        {
            // Translucent variant: glass refraction overrides the matte
            // baseline.  Note the per-instance FORCE_OPAQUE flag is
            // still set (the BLAS instance was built that way), so any-
            // hit doesn't fire here, but refraction (which doesn't need
            // any-hit, just back-face hits via RAY_FLAG_NONE on the
            // child trace) works fine.
            mat.refractivity = 0.85;
            mat.ior          = 1.50;
        }
        else
        {
            // Matte variant: brighten the cluster-tint so the matte
            // halves of the checker pattern read as VIBRANT colour
            // tiles, distinct from the translucent ones.
            mat.baseColor.xyz = saturate(mat.baseColor.xyz * 1.45);
        }
    }
    // Sphere1 (instanceID=1, the clear-glass-densest sphere) gets its
    // OWN checker pattern - alternate between SHINY MIRROR and the
    // baseline TRANSLUCENT glass.  Same parity-on-(lat+long) idea as
    // sphere2.  Sphere1 mesh: numLat=24 numLong=48 tileLat=4 tileLong=6
    // -> tilesLat=6, tilesLong=8 -> 48 clusters with IDs 100..147.
    if (InstanceID() == 1)
    {
        const uint kSphere1FirstCid = 100;
        const uint kSphere1TilesLong = 8;
        uint local   = cid - kSphere1FirstCid;
        uint latIdx  = local / kSphere1TilesLong;
        uint longIdx = local % kSphere1TilesLong;
        bool isB     = ((latIdx + longIdx) & 1u) != 0u;
        if (isB)
        {
            // Shiny mirror variant: mirror-opaque override.  Refraction
            // is killed (refr=0) so this cluster shows just the cluster-
            // tinted base color + a strong reflection.
            mat.reflectivity = 0.90;
            mat.refractivity = 0.0;
            mat.ior          = 0.0;
        }
        // else: keep the baseline translucent glass material.
    }
    // Floor (instanceID=6, the glass slab) gets a CHECKER too: alternate
    // top+bottom face TILES between SHINY MIRROR and TRANSLUCENT glass.
    // Slab layout (see GeneratePlaneSpatialTiles slab branch):
    //   600..635 : top    6x6 = 36 clusters
    //   636..671 : bottom 6x6 = 36 clusters (mirror of top, normal -Y)
    //   672..675 : 4 side walls - one big cluster spanning the whole edge
    // Same parity formula on (tu, tv) as the spheres so the top + bottom
    // checker stay aligned (a "tile" looks the same material from above
    // and below).  Side walls: remap to the adjacent top tile's cid using
    // the object-space hit position so the wall slice picks up the SAME
    // colour AND material as the floor tile it abuts (no visible seam).
    if (InstanceID() == 6)
    {
        const uint kFloorFirstCid = 600u;
        const uint kFloorTilesV   = 6u;
        const uint kTopBottomEnd  = kFloorFirstCid + 2u * 36u; // 672 = first wall

        // SIDE WALLS: remap cid -> adjacent top tile cid.
        if (cid >= 672u && cid <= 675u)
        {
            // Object-space hit position.  Slab is centered at origin in
            // object space, halfSize 3.5 in both X and Z, scale=1.0.
            float3 oPos = ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection();
            const float kHalf  = 3.5;
            const float kTileW = (2.0 * kHalf) / 6.0;          // ~1.167 units per tile
            int tu, tv;
            if (cid == 672u)        { tu = 0;                                                tv = clamp((int)floor((oPos.z + kHalf) / kTileW), 0, 5); }
            else if (cid == 673u)   { tu = 5;                                                tv = clamp((int)floor((oPos.z + kHalf) / kTileW), 0, 5); }
            else if (cid == 674u)   { tu = clamp((int)floor((oPos.x + kHalf) / kTileW), 0, 5); tv = 0;                                                }
            else /* cid == 675 */   { tu = clamp((int)floor((oPos.x + kHalf) / kTileW), 0, 5); tv = 5;                                                }
            cid = kFloorFirstCid + (uint)tu * kFloorTilesV + (uint)tv;
        }

        // BOTTOM TILES: remap cid -> matching top tile cid (subtract the
        // 36-tile offset) so the bottom face picks up the SAME cluster
        // colour as the top tile directly above it.  Without this remap
        // ClusterColor() hashes each cid independently and the bottom
        // tiles get a totally different palette - visually disorienting
        // when looking through a translucent top tile down to its
        // corresponding bottom (which the user would intuit as the
        // "same tile, other side").
        if (cid >= kFloorFirstCid + 36u && cid < kTopBottomEnd)
        {
            cid -= 36u;     // bottom (636..671) -> top (600..635)
        }

        if (cid < kTopBottomEnd)
        {
            // Local index within the top set (0..35) after the bottom-
            // and wall-remaps above.
            uint local   = (cid - kFloorFirstCid) % 36u;
            uint tu      = local / kFloorTilesV;
            uint tv      = local % kFloorTilesV;
            bool isB     = ((tu + tv) & 1u) != 0u;
            if (isB)
            {
                mat.reflectivity = 0.85;
                mat.refractivity = 0.0;
                mat.ior          = 0.0;
            }
            // else: keep the baseline translucent glass slab material.
        }
        // For BACK-FACE hits on the slab (ray is INSIDE the glass volume
        // hitting an interior surface from within - e.g. the camera ray
        // refracted through a translucent top tile and is now hitting
        // the matching bottom tile from inside), boost the surface
        // contribution by dropping refractivity.  Without this the
        // bottom face's cluster colour shows up as only ~14%% of the
        // exit pixel (mostly the sand beyond), so the user cannot SEE
        // the bottom face's checker pattern through the slab.  Cutting
        // refractivity to 40%% of baseline raises surface contribution
        // to ~50%% which lets the matching bottom-tile colour read
        // distinctly as an "inner glass layer" instead of just being
        // a tiny tint over sand.
        if (HitKind() == HIT_KIND_TRIANGLE_BACK_FACE && mat.refractivity > 0.0)
        {
            mat.refractivity *= 0.4;
        }
    }
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------

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
        // Tint the refracted RGB by the cluster colour so per-cluster
        // boundaries are VISIBLE through glass (acts like a stained-
        // glass filter - the cluster decomposition shows up as colour
        // variation in the refracted view, not just on the thin
        // surface contribution).  Half-strength relative to the surface
        // tint so the stained-glass effect is clearly visible without
        // drowning out what's seen THROUGH the glass.  Shared between
        // TIR and true-refraction branches below.
        float3 refractTint = lerp(float3(1, 1, 1), clusterCol, clusterTint * 0.50);

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
        // Cluster-tint strength on REFLECTIONS:
        //   chrome (sphere0) and most surfaces: 0.70 - high tint so the
        //     cluster grid reads on a near-perfect mirror (chrome would
        //     otherwise show only the mirrored scene, with no per-cluster
        //     visual differentiation).
        //   floor (instanceID=6): LOW tint (0.20) so the directional
        //     sky-gradient variation dominates - top tiles reflect
        //     vertically -> zenith deep blue, side tiles reflect
        //     horizontally -> horizon light blue + other objects.
        //     This is what makes the floor's top vs side visually
        //     distinct as MIRROR surfaces with the same cluster identity.
        float tintStrength = (InstanceID() == 6u) ? 0.20 : 0.70;
        float3 reflectTint = lerp(float3(1, 1, 1), clusterCol, tintStrength);
        finalColor = lerp(finalColor, reflectedRGB * reflectTint, F);
    }

    p.color = float4(finalColor, 1);
}
