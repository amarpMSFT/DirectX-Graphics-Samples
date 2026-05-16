//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// ProceduralGeometry.h
//
// Procedural mesh generators that decompose into "clusters" suitable for the
// DXR2 cluster build path. The decomposition strategy is **spatial tiling**:
// each cluster covers a small contiguous patch of the surface in UV space. This
// matches how engines (Nanite, NVIDIA's animated-clusters sample, etc.)
// decompose meshes for cluster acceleration structures - small spatially
// localized chunks rather than long thin strips.
//
// All meshes are emitted as a vector of self-contained clusters: each cluster
// owns its own vertex buffer (8-bit local indices into a per-cluster VB), so
// the encoder can encode each cluster independently and the cluster build args
// reference per-cluster VertexBuffer + IndexBuffer GPU VAs directly.
//
// Per-cluster vertex/triangle counts stay well within DXR2's 256/256 limit
// for the parameter ranges this header is used at.
//
#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ProceduralGeometry
{
    struct float3 { float x, y, z; };

    struct Cluster
    {
        unsigned int            clusterID;     // exposed via HLSL ClusterID() in hits
        std::vector<float3>     positions;     // local positions (mesh-local space)
        std::vector<uint8_t>    indices;       // local 8-bit indices into positions
    };

    struct Mesh
    {
        std::vector<Cluster>    clusters;
        unsigned int            totalTriangles = 0;
        unsigned int            totalVertices  = 0;
    };

    // ------------------------------------------------------------------
    // UV sphere with **spatial-tile** cluster decomposition.
    // Generates a (numLat x numLong) quad mesh, then groups quads into
    // tiles of `tileLatSize` rows × `tileLongSize` columns, one cluster
    // per tile. Each tile owns its own vertex buffer with the
    // (tileLatSize+1) × (tileLongSize+1) vertices it needs.
    // numLat must be divisible by tileLatSize; numLong by tileLongSize.
    // ------------------------------------------------------------------
    inline Mesh GenerateUVSphereSpatialTiles(
        float radius, int numLat, int numLong,
        int tileLatSize, int tileLongSize, unsigned int firstClusterID = 0)
    {
        Mesh m;
        const int tilesLat  = numLat  / tileLatSize;
        const int tilesLong = numLong / tileLongSize;
        m.clusters.reserve((size_t)tilesLat * tilesLong);

        unsigned int clusterCounter = firstClusterID;
        for (int tLat = 0; tLat < tilesLat; ++tLat)
        for (int tLong = 0; tLong < tilesLong; ++tLong)
        {
            Cluster c;
            c.clusterID = clusterCounter++;

            // Latitude/longitude index ranges this tile covers.
            const int latLo  = tLat  * tileLatSize;
            const int longLo = tLong * tileLongSize;

            // Vertex grid: (tileLatSize+1) rows × (tileLongSize+1) cols.
            const int rowSize = tileLongSize + 1;
            for (int li = 0; li <= tileLatSize; ++li)
            for (int lj = 0; lj <= tileLongSize; ++lj)
            {
                int latIdx  = latLo + li;
                int longIdx = longLo + lj;
                // Wrap longitude to share vertices across tile-east boundary,
                // even though we duplicate them within the cluster's local VB.
                // (Wrap matters for the global-equality story: vertices on the
                // shared edge get identical input positions and, with shared
                // exponent, identical decoded positions -> watertight.)
                if (longIdx == numLong) longIdx = 0;

                float theta = float((double)latIdx  * M_PI       / numLat);
                float phi   = float((double)longIdx * 2.0 * M_PI / numLong);
                float sinT = std::sin(theta), cosT = std::cos(theta);
                float cosP = std::cos(phi),   sinP = std::sin(phi);
                c.positions.push_back({ radius * sinT * cosP,
                                        radius * cosT,
                                        radius * sinT * sinP });
            }

            // Two CCW (viewed from outside) triangles per quad.
            for (int li = 0; li < tileLatSize; ++li)
            for (int lj = 0; lj < tileLongSize; ++lj)
            {
                uint8_t i00 = (uint8_t)(li     * rowSize + lj    );  // top-left
                uint8_t i01 = (uint8_t)(li     * rowSize + lj + 1);  // top-right
                uint8_t i10 = (uint8_t)((li+1) * rowSize + lj    );  // bot-left
                uint8_t i11 = (uint8_t)((li+1) * rowSize + lj + 1);  // bot-right
                c.indices.push_back(i00); c.indices.push_back(i11); c.indices.push_back(i10);
                c.indices.push_back(i00); c.indices.push_back(i01); c.indices.push_back(i11);
            }

            m.totalTriangles += (unsigned int)tileLatSize * tileLongSize * 2;
            m.totalVertices  += (unsigned int)(tileLatSize + 1) * (tileLongSize + 1);
            m.clusters.push_back(std::move(c));
        }

        return m;
    }

    // ------------------------------------------------------------------
    // Torus with spatial-tile cluster decomposition.
    // ringSegs   = segments around the major (large) ring
    // sideSegs   = segments around the minor (tube cross-section) ring
    // tileRing/Side same divisibility constraints as the sphere version.
    // ------------------------------------------------------------------
    inline Mesh GenerateTorusSpatialTiles(
        float majorRadius, float minorRadius,
        int ringSegs, int sideSegs,
        int tileRing, int tileSide, unsigned int firstClusterID = 0)
    {
        Mesh m;
        const int tilesRing = ringSegs / tileRing;
        const int tilesSide = sideSegs / tileSide;
        m.clusters.reserve((size_t)tilesRing * tilesSide);

        unsigned int clusterCounter = firstClusterID;
        for (int tR = 0; tR < tilesRing; ++tR)
        for (int tS = 0; tS < tilesSide; ++tS)
        {
            Cluster c;
            c.clusterID = clusterCounter++;
            const int rLo = tR * tileRing;
            const int sLo = tS * tileSide;

            const int rowSize = tileSide + 1;
            for (int ri = 0; ri <= tileRing; ++ri)
            for (int si = 0; si <= tileSide; ++si)
            {
                int rIdx = rLo + ri;
                int sIdx = sLo + si;
                if (rIdx == ringSegs) rIdx = 0;
                if (sIdx == sideSegs) sIdx = 0;

                float u = float((double)rIdx * 2.0 * M_PI / ringSegs);
                float v = float((double)sIdx * 2.0 * M_PI / sideSegs);
                float cu = std::cos(u), su = std::sin(u);
                float cv = std::cos(v), sv = std::sin(v);
                float r  = majorRadius + minorRadius * cv;
                c.positions.push_back({ r * cu, minorRadius * sv, r * su });
            }

            for (int ri = 0; ri < tileRing; ++ri)
            for (int si = 0; si < tileSide; ++si)
            {
                uint8_t i00 = (uint8_t)(ri     * rowSize + si    );
                uint8_t i01 = (uint8_t)(ri     * rowSize + si + 1);
                uint8_t i10 = (uint8_t)((ri+1) * rowSize + si    );
                uint8_t i11 = (uint8_t)((ri+1) * rowSize + si + 1);
                c.indices.push_back(i00); c.indices.push_back(i11); c.indices.push_back(i10);
                c.indices.push_back(i00); c.indices.push_back(i01); c.indices.push_back(i11);
            }

            m.totalTriangles += (unsigned int)tileRing * tileSide * 2;
            m.totalVertices  += (unsigned int)(tileRing + 1) * (tileSide + 1);
            m.clusters.push_back(std::move(c));
        }
        return m;
    }

    // ------------------------------------------------------------------
    // Subdivided cube ("rounded box" without the rounding). Each face is
    // tessellated into (faceSubdiv x faceSubdiv) quads, each face becomes
    // (faceSubdiv/tileSize)^2 spatial-tile clusters. faceSubdiv must be
    // divisible by tileSize.
    // ------------------------------------------------------------------
    inline Mesh GenerateCubeSpatialTiles(
        float halfSize, int faceSubdiv, int tileSize, unsigned int firstClusterID = 0)
    {
        Mesh m;
        // Face basis: list of (origin, axisU, axisV) for each of the 6 faces.
        // axisU x axisV must equal the OUTWARD normal so the resulting
        // triangles have the correct CCW-from-outside winding for DXR's
        // RAY_FLAG_CULL_BACK_FACING_TRIANGLES.
        struct Face { float3 origin, axisU, axisV; };
        const float h = halfSize;
        const Face faces[6] = {
            // +X face (normal +X): U=+Y, V=+Z, U x V = +X
            {{ +h, -h, -h}, {0, 1, 0}, {0, 0, 1}},
            // -X face (normal -X): U=+Z, V=+Y, U x V = -X
            {{ -h, -h, -h}, {0, 0, 1}, {0, 1, 0}},
            // +Y face (normal +Y): U=+Z, V=+X, U x V = +Y
            {{ -h, +h, -h}, {0, 0, 1}, {1, 0, 0}},
            // -Y face (normal -Y): U=+X, V=+Z, U x V = -Y
            {{ -h, -h, -h}, {1, 0, 0}, {0, 0, 1}},
            // +Z face (normal +Z): U=+X, V=+Y, U x V = +Z
            {{ -h, -h, +h}, {1, 0, 0}, {0, 1, 0}},
            // -Z face (normal -Z): U=+Y, V=+X, U x V = -Z
            {{ -h, -h, -h}, {0, 1, 0}, {1, 0, 0}},
        };

        const int tilesPerSide = faceSubdiv / tileSize;
        m.clusters.reserve((size_t)6 * tilesPerSide * tilesPerSide);

        unsigned int clusterCounter = firstClusterID;
        for (int f = 0; f < 6; ++f)
        for (int tu = 0; tu < tilesPerSide; ++tu)
        for (int tv = 0; tv < tilesPerSide; ++tv)
        {
            Cluster c;
            c.clusterID = clusterCounter++;
            const Face& face = faces[f];
            const float du = (2 * h) / faceSubdiv;
            const int rowSize = tileSize + 1;
            for (int li = 0; li <= tileSize; ++li)
            for (int lj = 0; lj <= tileSize; ++lj)
            {
                float u = du * (tu * tileSize + lj);
                float v = du * (tv * tileSize + li);
                c.positions.push_back({
                    face.origin.x + face.axisU.x * u + face.axisV.x * v,
                    face.origin.y + face.axisU.y * u + face.axisV.y * v,
                    face.origin.z + face.axisU.z * u + face.axisV.z * v });
            }
            for (int li = 0; li < tileSize; ++li)
            for (int lj = 0; lj < tileSize; ++lj)
            {
                uint8_t i00 = (uint8_t)(li     * rowSize + lj    );
                uint8_t i01 = (uint8_t)(li     * rowSize + lj + 1);
                uint8_t i10 = (uint8_t)((li+1) * rowSize + lj    );
                uint8_t i11 = (uint8_t)((li+1) * rowSize + lj + 1);
                c.indices.push_back(i00); c.indices.push_back(i11); c.indices.push_back(i10);
                c.indices.push_back(i00); c.indices.push_back(i01); c.indices.push_back(i11);
            }
            m.totalTriangles += (unsigned int)tileSize * tileSize * 2;
            m.totalVertices  += (unsigned int)(tileSize + 1) * (tileSize + 1);
            m.clusters.push_back(std::move(c));
        }
        return m;
    }

    // ------------------------------------------------------------------
    // Flat XZ-plane built as a (tilesU x tilesV) grid of cluster tiles,
    // each cluster being (tileQuadsU x tileQuadsV) quads in the plane's
    // local frame (centred at origin, +Y normal). The N-cluster floor
    // demonstrates that cluster geometry doesn't have to be curved -
    // a flat ground plane benefits from CLAS just as much (still gets
    // per-cluster opacity flags, BLAS-from-CLAS aggregation, and the
    // memory savings of compressed1 / position-truncate).
    //
    // Output verts per cluster = (tileQuadsU+1) * (tileQuadsV+1); keep
    // the product <= 256 so 8-bit indices remain valid.
    // ------------------------------------------------------------------
    inline Mesh GeneratePlaneSpatialTiles(
        float halfSizeU, float halfSizeV,
        int tilesU, int tilesV,
        int tileQuadsU, int tileQuadsV,
        unsigned int firstClusterID = 0)
    {
        Mesh m;
        m.clusters.reserve((size_t)tilesU * tilesV);
        const int rowSize = tileQuadsU + 1;
        const float tileWidthU = 2.0f * halfSizeU / (float)tilesU;
        const float tileWidthV = 2.0f * halfSizeV / (float)tilesV;
        const float quadWidthU = tileWidthU / (float)tileQuadsU;
        const float quadWidthV = tileWidthV / (float)tileQuadsV;

        unsigned int clusterCounter = firstClusterID;
        for (int tu = 0; tu < tilesU; ++tu)
        for (int tv = 0; tv < tilesV; ++tv)
        {
            Cluster c;
            c.clusterID = clusterCounter++;
            const float baseU = -halfSizeU + (float)tu * tileWidthU;
            const float baseV = -halfSizeV + (float)tv * tileWidthV;

            // Vertices: (tileQuadsU+1) x (tileQuadsV+1) on the XZ plane (Y=0).
            for (int li = 0; li <= tileQuadsU; ++li)
            for (int lj = 0; lj <= tileQuadsV; ++lj)
            {
                c.positions.push_back({
                    baseU + (float)li * quadWidthU,
                    0.0f,
                    baseV + (float)lj * quadWidthV
                });
            }
            // Indices: two triangles per quad, both wound CCW when viewed from
            // above (i.e. normal = +Y). Camera sits at +Y > 0 so it looks
            // DOWN at the floor; without the +Y winding, RAY_FLAG_CULL_BACK_-
            // FACING_TRIANGLES would silently hide the entire floor.
            for (int li = 0; li < tileQuadsU; ++li)
            for (int lj = 0; lj < tileQuadsV; ++lj)
            {
                uint8_t i00 = (uint8_t)(li     * rowSize + lj    );
                uint8_t i01 = (uint8_t)(li     * rowSize + lj + 1);
                uint8_t i10 = (uint8_t)((li+1) * rowSize + lj    );
                uint8_t i11 = (uint8_t)((li+1) * rowSize + lj + 1);
                c.indices.push_back(i00); c.indices.push_back(i01); c.indices.push_back(i11);
                c.indices.push_back(i00); c.indices.push_back(i11); c.indices.push_back(i10);
            }
            m.totalTriangles += (unsigned int)tileQuadsU * tileQuadsV * 2;
            m.totalVertices  += (unsigned int)(tileQuadsU + 1) * (tileQuadsV + 1);
            m.clusters.push_back(std::move(c));
        }
        return m;
    }

    // ------------------------------------------------------------------
    // Klein bottle - figure-8 immersion in R^3.  A non-orientable closed
    // surface (no inside/outside!), parametrised over (u,v) in [0, 2π]^2
    // by:
    //
    //     half = u/2
    //     r = a + cos(half)*sin(v) - sin(half)*sin(2v)
    //     x = r * cos(u)
    //     y = sin(half)*sin(v) + cos(half)*sin(2v)
    //     z = r * sin(u)
    //
    // (Y up; the standard Klein-bottle "twisted donut" silhouette.)
    //
    // The figure-8 immersion DOES self-intersect in 3-space (an unavoidable
    // consequence of squeezing a non-orientable surface into 3D), which is
    // genuinely useful for translucency demos: refractive / stochastic rays
    // pass into and out of multiple surface sheets, producing strikingly
    // animated distortions and frosted-glass overlap.
    //
    // Mesh is decomposed into (tilesU x tilesV) cluster tiles in the same
    // way as GenerateTorusSpatialTiles - one cluster per (tileU, tileV)
    // patch, each cluster owning its own (tileQuadsU+1) x (tileQuadsV+1)
    // local vertex buffer.
    // ------------------------------------------------------------------
    inline Mesh GenerateKleinBottleSpatialTiles(
        float bottleScale,
        int numU, int numV,
        int tileUSize, int tileVSize,
        unsigned int firstClusterID = 0)
    {
        Mesh m;
        const int tilesU = numU / tileUSize;
        const int tilesV = numV / tileVSize;
        m.clusters.reserve((size_t)tilesU * tilesV);
        const float kPi = 3.14159265358979323846f;
        const float a   = 2.0f;            // figure-8 main radius

        unsigned int clusterCounter = firstClusterID;
        for (int tu = 0; tu < tilesU; ++tu)
        for (int tv = 0; tv < tilesV; ++tv)
        {
            Cluster c;
            c.clusterID = clusterCounter++;
            const int rowSize = tileVSize + 1;

            // Vertices on a (tileUSize+1) x (tileVSize+1) grid in (u,v)
            // parameter space.
            for (int li = 0; li <= tileUSize; ++li)
            for (int lj = 0; lj <= tileVSize; ++lj)
            {
                const int gi = tu * tileUSize + li;
                const int gj = tv * tileVSize + lj;
                const float u = (float)gi / (float)numU * 2.0f * kPi;
                const float v = (float)gj / (float)numV * 2.0f * kPi;
                const float half = u * 0.5f;
                const float ch = std::cos(half), sh = std::sin(half);
                const float cv = std::cos(v),    sv = std::sin(v);
                const float s2v = std::sin(2.0f * v);
                const float r   = a + ch * sv - sh * s2v;
                const float x   = r * std::cos(u);
                const float y   = sh * sv + ch * s2v;
                const float z   = r * std::sin(u);
                c.positions.push_back({ bottleScale * x,
                                        bottleScale * y,
                                        bottleScale * z });
            }
            // Indices: two CCW triangles per quad (winding consistent with
            // the parametric surface's natural normal direction).
            for (int li = 0; li < tileUSize; ++li)
            for (int lj = 0; lj < tileVSize; ++lj)
            {
                uint8_t i00 = (uint8_t)(li     * rowSize + lj    );
                uint8_t i01 = (uint8_t)(li     * rowSize + lj + 1);
                uint8_t i10 = (uint8_t)((li+1) * rowSize + lj    );
                uint8_t i11 = (uint8_t)((li+1) * rowSize + lj + 1);
                c.indices.push_back(i00); c.indices.push_back(i11); c.indices.push_back(i10);
                c.indices.push_back(i00); c.indices.push_back(i01); c.indices.push_back(i11);
            }
            m.totalTriangles += (unsigned int)tileUSize * tileVSize * 2;
            m.totalVertices  += (unsigned int)(tileUSize + 1) * (tileVSize + 1);
            m.clusters.push_back(std::move(c));
        }
        return m;
    }
}
