//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// ProceduralGeometry.h
//
// Minimal procedural mesh generators used by the Partitioned TLAS sample.
// Keep this self-contained and header-only -- mesh-gen isn't the point of the
// sample, the meshes just give PTLAS something to instance.
//
// Generators:
//   * MakeIcosphere(subdivisions) -- unit sphere centered at origin
//   * MakeTorus    (major, minor, segMajor, segMinor) -- torus around Y axis
//
// Both return tightly-packed float3 positions + uint32 indices.  Normals are
// derived in the closest-hit shader from the triangle's barycentric position
// + world geometry, so no per-vertex normals are needed.
//
#pragma once

#include <vector>
#include <cstdint>
#include <DirectXMath.h>
#include <unordered_map>
#include <cmath>

namespace ProceduralGeometry
{
    using DirectX::XMFLOAT3;

    struct Mesh
    {
        std::vector<XMFLOAT3>   positions;
        std::vector<uint32_t>   indices;        // 3 per triangle

        uint32_t TriangleCount() const { return (uint32_t)(indices.size() / 3u); }
        uint32_t VertexCount()   const { return (uint32_t)positions.size(); }
    };

    // -----------------------------------------------------------------------
    // Icosphere: start from a regular icosahedron, subdivide each triangle
    // into 4 (Loop subdivision pattern) `subdiv` times, project all vertices
    // to the unit sphere.  Edge-midpoint cache de-duplicates split vertices.
    //   subdiv=0 -> 12 verts,  20 tris
    //   subdiv=1 -> 42 verts,  80 tris
    //   subdiv=2 -> 162 verts, 320 tris
    //   subdiv=3 -> 642 verts, 1280 tris
    // -----------------------------------------------------------------------
    inline Mesh MakeIcosphere(uint32_t subdiv)
    {
        Mesh m;

        // 12 base icosahedron vertices (golden-ratio cross product trick).
        const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;   // golden ratio
        const XMFLOAT3 base[12] = {
            { -1,  t,  0 }, {  1,  t,  0 }, { -1, -t,  0 }, {  1, -t,  0 },
            {  0, -1,  t }, {  0,  1,  t }, {  0, -1, -t }, {  0,  1, -t },
            {  t,  0, -1 }, {  t,  0,  1 }, { -t,  0, -1 }, { -t,  0,  1 },
        };
        const uint32_t baseIdx[20 * 3] = {
            0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
            1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
            3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
            4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
        };

        auto normalize = [](XMFLOAT3 v) -> XMFLOAT3 {
            const float r = 1.0f / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
            return { v.x*r, v.y*r, v.z*r };
        };

        m.positions.reserve(12);
        for (auto& p : base) m.positions.push_back(normalize(p));
        m.indices.assign(baseIdx, baseIdx + 60);

        // Midpoint cache: key = packed (loVert, hiVert) edge -> midpoint vertex
        // index.  Prevents duplicating shared edge midpoints across two
        // neighbour triangles, which would also unbind the sphere topology.
        auto packKey = [](uint32_t a, uint32_t b) -> uint64_t {
            if (a > b) std::swap(a, b);
            return (uint64_t(a) << 32) | uint64_t(b);
        };

        for (uint32_t s = 0; s < subdiv; ++s)
        {
            std::unordered_map<uint64_t, uint32_t> midCache;
            std::vector<uint32_t> nextIdx;
            nextIdx.reserve(m.indices.size() * 4);

            auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
                uint64_t k = packKey(a, b);
                auto it = midCache.find(k);
                if (it != midCache.end()) return it->second;
                const XMFLOAT3& pa = m.positions[a];
                const XMFLOAT3& pb = m.positions[b];
                uint32_t idx = (uint32_t)m.positions.size();
                m.positions.push_back(normalize({
                    (pa.x + pb.x) * 0.5f,
                    (pa.y + pb.y) * 0.5f,
                    (pa.z + pb.z) * 0.5f }));
                midCache.emplace(k, idx);
                return idx;
            };

            for (size_t i = 0; i < m.indices.size(); i += 3)
            {
                uint32_t a = m.indices[i + 0];
                uint32_t b = m.indices[i + 1];
                uint32_t c = m.indices[i + 2];
                uint32_t ab = midpoint(a, b);
                uint32_t bc = midpoint(b, c);
                uint32_t ca = midpoint(c, a);
                nextIdx.insert(nextIdx.end(), {
                    a, ab, ca,
                    b, bc, ab,
                    c, ca, bc,
                    ab, bc, ca,
                });
            }
            m.indices = std::move(nextIdx);
        }

        return m;
    }

    // -----------------------------------------------------------------------
    // Torus around Y axis.  majorR = ring radius (center to tube center),
    // minorR = tube radius.  segMajor / segMinor = subdivision counts.
    // -----------------------------------------------------------------------
    inline Mesh MakeTorus(float majorR, float minorR, uint32_t segMajor, uint32_t segMinor)
    {
        Mesh m;
        m.positions.reserve(segMajor * segMinor);
        m.indices.reserve(segMajor * segMinor * 6);

        const float kTwoPi = 6.28318530718f;
        for (uint32_t u = 0; u < segMajor; ++u)
        {
            const float a = (float)u * kTwoPi / (float)segMajor;
            const float ca = std::cos(a), sa = std::sin(a);
            for (uint32_t v = 0; v < segMinor; ++v)
            {
                const float b = (float)v * kTwoPi / (float)segMinor;
                const float cb = std::cos(b), sb = std::sin(b);
                const float r = majorR + minorR * cb;
                m.positions.push_back({ r * ca, minorR * sb, r * sa });
            }
        }
        for (uint32_t u = 0; u < segMajor; ++u)
        {
            uint32_t un = (u + 1) % segMajor;
            for (uint32_t v = 0; v < segMinor; ++v)
            {
                uint32_t vn = (v + 1) % segMinor;
                uint32_t i00 = u  * segMinor + v;
                uint32_t i10 = un * segMinor + v;
                uint32_t i01 = u  * segMinor + vn;
                uint32_t i11 = un * segMinor + vn;
                // Winding reversed vs. {i00,i10,i11,i00,i11,i01} so the
                // cross product (e1×e2) points OUTWARD (away from torus
                // surface) matching the icosphere convention.  Without this
                // reversal the torus's "front" face is the INSIDE -- which
                // back-face culling then hides from external view.
                m.indices.insert(m.indices.end(), { i00, i11, i10, i00, i01, i11 });
            }
        }
        return m;
    }
}
