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

    // CLUSTER_FLAG_INTERIOR_SURFACE: this cluster is a face of an enclosed
    // glass volume (e.g. the slab's bottom face seen from inside, or one
    // of the slab's side walls).  The shader uses this to apply a back-
    // face refractivity boost so the interior surface reads as visible
    // instead of washing out the camera ray with transmitted exterior.
    static constexpr unsigned int CLUSTER_FLAG_INTERIOR_SURFACE = 0x1u;

    struct Cluster
    {
        unsigned int            clusterID;     // exposed via HLSL ClusterID() in hits
        std::vector<float3>     positions;     // local positions (mesh-local space)
        std::vector<float3>     normals;       // parallel to positions; analytic per-vertex
                                               // surface normals.  Travels through a SEPARATE
                                               // structured-buffer side channel because
                                               // DXR2 cluster geometry's CLAS only takes
                                               // positions in its vertex buffer.
        std::vector<uint8_t>    indices;       // local 8-bit indices into positions/normals

        // ------------------------------------------------------------------
        // Per-cluster GENERIC metadata, set by each generator at construction
        // time.  Lets the scene-build code drive per-cluster material /
        // colour overrides UNIFORMLY across all object types - no special-
        // casing per-instance in the closesthit shader, no hand-mirrored
        // cluster-id ranges in CPU code either.  Generic checker config
        // can be applied as `isB = (gridU + gridV) & 1` regardless of
        // whether this is a sphere lat/long tile, a slab top tile, or a
        // slab wall sub-cluster.
        // ------------------------------------------------------------------
        unsigned int            gridU = 0;         // 1st grid coord (latIdx / tu)
        unsigned int            gridV = 0;         // 2nd grid coord (longIdx / tv)
        unsigned int            matchedColorCid;   // ClusterColor() hash key.  For most
                                                   // clusters this == clusterID, but for
                                                   // slab bottom + wall sub-clusters the
                                                   // generator sets this to the matching
                                                   // TOP tile's cid so the colour layer
                                                   // is unified across the slab volume.
        unsigned int            flags = 0;         // CLUSTER_FLAG_* bits.
        // ------------------------------------------------------------------
        // Material region index within the owning OBJECT.  Lets a single
        // object split into multiple material regions: clusters with
        // matRegionIdx=0 form geometry 0 of the object's BLAS, clusters
        // with matRegionIdx=1 form geometry 1, etc.  Used by both paths:
        //   Cluster:     BUILD_CLAS_FROM_TRIANGLES_ARGS::BaseGeometryIndex
        //                gets stamped with this value per CLAS, so the
        //                BLAS-from-CLAS has geometries grouped by region
        //                and GeometryIndex() in the closest-hit returns
        //                this value at hit time.
        //   Traditional: clusters are reordered within each object so
        //                same-region clusters are contiguous in the VB+IB,
        //                and one D3D12_RAYTRACING_GEOMETRY_DESC is emitted
        //                per region (covering all that region's clusters).
        // Default 0 (single-region object) is what every procedural
        // generator emits; scene code overrides for multi-material objects
        // like the mixed-material small sphere.
        unsigned int            matRegionIdx = 0;
    };

    struct Mesh
    {
        std::vector<Cluster>    clusters;
        unsigned int            totalTriangles = 0;
        unsigned int            totalVertices  = 0;
    };



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
            c.gridU = (unsigned int)tu;
            c.gridV = (unsigned int)tv;
            c.matchedColorCid = c.clusterID;
            const Face& face = faces[f];
            // Per-face flat normal = axisU x axisV (outward by construction
            // of the face table).  All vertices on this cluster share it -
            // cubes don't smooth across edges.
            const float3 nFace = {
                face.axisU.y * face.axisV.z - face.axisU.z * face.axisV.y,
                face.axisU.z * face.axisV.x - face.axisU.x * face.axisV.z,
                face.axisU.x * face.axisV.y - face.axisU.y * face.axisV.x };
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
                c.normals.push_back(nFace);
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


}

