//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// FillClasFromTrianglesArgs.hlsl
//
// GPU-side generator for the per-cluster
// D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS array (the args
// the BUILD_CLAS pass reads).  CPU prepares a small per-cluster metadata
// buffer + a few global constants; this CS expands each metadata entry
// into the full 80-byte args struct.
//
// Per-cluster metadata layout (28 bytes/cluster, matches ClasArgsMeta in
// the C++ side):
//
//   offset  0 (4 B)   UINT   clusterID
//   offset  4 (4 B)   UINT   triangleCount   (must fit in u16)
//   offset  8 (4 B)   UINT   vertexCount     (must fit in u16)
//   offset 12 (4 B)   UINT   vbByteOffset    (from g_baseGpuVa)
//   offset 16 (4 B)   UINT   ibByteOffset    (from g_baseGpuVa)
//   offset 20 (4 B)   UINT   opaqueFlag      (0 or D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE)
//   offset 24 (4 B)   UINT   matRegionIdx    (becomes per-CLAS BaseGeometryIndex)

#define HLSL
#include "RaytracingHlslCompat.h"

cbuffer Constants : register(b0)
{
    uint g_baseGpuVaLo;
    uint g_baseGpuVaHi;
    uint g_clusterCount;
    uint g_vertexBufferStride;     // 12 (FLOAT32_3) or 0 (COMPRESSED1)
    uint g_positionTruncateBits;   // 0..23
};

ByteAddressBuffer   g_meta    : register(t0);
RWByteAddressBuffer g_argsOut : register(u0);

uint2 add64(uint gvaLo, uint gvaHi, uint offset)
{
    const uint sumLo = gvaLo + offset;
    const uint carry = (sumLo < gvaLo) ? 1u : 0u;
    return uint2(sumLo, gvaHi + carry);
}
uint pack16(uint lo, uint hi) { return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16); }

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint idx = tid.x;
    if (idx >= g_clusterCount) return;

    // Load 28 bytes of per-cluster metadata.
    const uint3 m0 = g_meta.Load3(idx * 28u +  0);  // {clusterID, triCount, vertCount}
    const uint3 m1 = g_meta.Load3(idx * 28u + 12);  // {vbOff, ibOff, opaqueFlag}
    const uint  m2 = g_meta.Load (idx * 28u + 24);  // {matRegionIdx}

    const uint  clusterID    = m0.x;
    const uint  triCount     = m0.y;
    const uint  vertCount    = m0.z;
    const uint  vbOff        = m1.x;
    const uint  ibOff        = m1.y;
    const uint  opaqueFlag   = m1.z;
    const uint  matRegionIdx = m2;

    const uint2 vbGva = add64(g_baseGpuVaLo, g_baseGpuVaHi, vbOff);
    const uint2 ibGva = add64(g_baseGpuVaLo, g_baseGpuVaHi, ibOff);

    // BaseGeometryIndexAndFlags per Raytracing2.md:
    //   bits 23:0  = Geometry Index   (LOW 24 bits)
    //   bits 31:24 = flags            (HIGH 8 bits) -- FLAG_OPAQUE = 0x80000000
    const uint baseGeomIdxAndFlags = (matRegionIdx & 0x00FFFFFFu) | (opaqueFlag & 0xFF000000u);

    const uint baseByte = idx * 80u;
    g_argsOut.Store (baseByte +  0, clusterID);
    g_argsOut.Store (baseByte +  4, 0u);                                            // ClusterFlags
    g_argsOut.Store (baseByte +  8, pack16(triCount, vertCount));
    g_argsOut.Store (baseByte + 12, baseGeomIdxAndFlags);
    g_argsOut.Store (baseByte + 16, 0u);                                            // OpacityMicromapBaseLocation
    g_argsOut.Store (baseByte + 20, pack16(g_vertexBufferStride, 2u));              // VBStride/IBStride (UINT16 indices = 2)
    g_argsOut.Store (baseByte + 24, pack16(0u, 0u));
    g_argsOut.Store (baseByte + 28, pack16(g_positionTruncateBits, 0u));
    g_argsOut.Store2(baseByte + 32, vbGva);
    g_argsOut.Store2(baseByte + 40, ibGva);
    g_argsOut.Store2(baseByte + 48, uint2(0, 0));
    g_argsOut.Store2(baseByte + 56, uint2(0, 0));
    g_argsOut.Store2(baseByte + 64, uint2(0, 0));
    g_argsOut.Store2(baseByte + 72, uint2(0, 0));
}
