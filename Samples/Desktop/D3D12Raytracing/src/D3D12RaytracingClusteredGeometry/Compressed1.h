//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// Compressed1.h
//
// CPU-side encoder for the DXR2 D3D12_VERTEX_FORMAT_COMPRESSED1 vertex format.
//
// Per the spec (Raytracing2.md "Compressed1 position encoding"), each cluster's
// vertex data is laid out as:
//   - 12-byte D3D12_VERTEX_FORMAT_COMPRESSED1_HEADER (exponent, 24-bit signed
//     anchor xyz, 4-bit-each x/y/z bit counts in [1..16])
//   - immediately followed by per-vertex (x_bits + y_bits + z_bits)-bit packed
//     offsets (no padding between vertices)
//
// Decoding:  pos[c] = (anchor[c] + offset[c]) * 2^(exponent - 127)
//
// This implementation uses a single bit width N for all three components and
// picks the exponent automatically so the per-component dynamic range fits in
// 2^N - 1 levels with the smallest sufficient unit. That is the encoding most
// engines will pick for static geometry (the spec's recommended best practice
// for this sample).
//
// We use the d3d12.h ENCODE_/DECODE_ macros to set/get header bits so any
// future spec tweaks pick up automatically.
//
#pragma once

#include "ProceduralGeometry.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <d3d12.h>

namespace Compressed1
{
    struct EncodedCluster
    {
        D3D12_VERTEX_FORMAT_COMPRESSED1_HEADER header = {};
        std::vector<uint8_t> bitstream;      // packed offsets (no padding)
        unsigned int          vertexCount   = 0;
        unsigned int          bitsPerComp   = 0;

        // Total bytes for the whole compressed cluster (header + bitstream).
        size_t TotalBytes() const { return sizeof(header) + bitstream.size(); }
    };

    // Pack least-significant `nBits` of `value` starting at bit `bitOffset` in `dst`.
    inline void WriteBits(std::vector<uint8_t>& dst, uint64_t bitOffset, uint32_t value, int nBits)
    {
        for (int b = 0; b < nBits; ++b)
        {
            uint64_t p = bitOffset + b;
            if ((value >> b) & 1u)
                dst[(size_t)(p / 8)] |= (uint8_t)(1u << (p % 8));
        }
    }

    inline uint32_t ReadBits(const std::vector<uint8_t>& src, uint64_t bitOffset, int nBits)
    {
        uint32_t v = 0;
        for (int b = 0; b < nBits; ++b)
        {
            uint64_t p = bitOffset + b;
            if (src[(size_t)(p / 8)] & (uint8_t)(1u << (p % 8))) v |= (1u << b);
        }
        return v;
    }

    // Encode `verts` into compressed1 with `nBitsPerComponent` bits per axis.
    // If `forcedExponent < 0`, the exponent is auto-picked per-cluster from the
    // cluster's own dynamic range. If `forcedExponent` is in [1..232], that
    // BIASED exponent is used directly (use this to keep adjacent clusters in a
    // shared encoding grid -> watertight at cluster boundaries).
    inline EncodedCluster Encode(const std::vector<ProceduralGeometry::float3>& verts,
                                 int nBitsPerComponent,
                                 int forcedExponent = -1)
    {
        EncodedCluster out;
        out.vertexCount = (unsigned int)verts.size();
        out.bitsPerComp = (unsigned int)nBitsPerComponent;

        if (verts.empty()) return out;

        // 1) per-axis min/max in input units
        float mn[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX };
        float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (auto& v : verts)
        {
            const float p[3] = { v.x, v.y, v.z };
            for (int c = 0; c < 3; ++c)
            {
                if (p[c] < mn[c]) mn[c] = p[c];
                if (p[c] > mx[c]) mx[c] = p[c];
            }
        }

        // 2) pick the unit so that the largest per-axis extent fits in (2^N - 1)
        //    quantization levels with a power-of-two unit.
        int e;
        if (forcedExponent > 0 && forcedExponent <= 232)
        {
            e = forcedExponent;
        }
        else
        {
            float maxExtent = 0.f;
            for (int c = 0; c < 3; ++c) maxExtent = std::max(maxExtent, mx[c] - mn[c]);
            const float maxLevels = float((1ull << nBitsPerComponent) - 1);
            const float minUnit   = (maxExtent > 0.f) ? (maxExtent / maxLevels) : 1e-30f;
            e = (int)std::ceil(std::log2(minUnit)) + 127;
            if (e < 1)   e = 1;
            if (e > 232) e = 232;
        }
        const float unit = std::ldexp(1.0f, e - 127);

        // 3) quantize, derive anchor as the per-axis quantized minimum
        std::vector<int32_t> qX(verts.size()), qY(verts.size()), qZ(verts.size());
        for (size_t i = 0; i < verts.size(); ++i)
        {
            qX[i] = (int32_t)std::lround(verts[i].x / unit);
            qY[i] = (int32_t)std::lround(verts[i].y / unit);
            qZ[i] = (int32_t)std::lround(verts[i].z / unit);
        }
        const int32_t aX = *std::min_element(qX.begin(), qX.end());
        const int32_t aY = *std::min_element(qY.begin(), qY.end());
        const int32_t aZ = *std::min_element(qZ.begin(), qZ.end());

        // Sanity: anchors must fit a 24-bit signed range, offsets must fit nBits.
        const int32_t kMin24 = -(1 << 23);
        const int32_t kMax24 =  (1 << 23) - 1;
        (void)kMin24; (void)kMax24;
        const uint32_t maxOff = (1u << nBitsPerComponent) - 1;

        // 4) header
        ENCODE_D3D12_COMPRESSED1(out.header,
                                 e,
                                 aX, aY, aZ,
                                 nBitsPerComponent - 1,
                                 nBitsPerComponent - 1,
                                 nBitsPerComponent - 1);

        // 5) bitstream (3*N bits per vertex, packed)
        const uint64_t totalBits = 3ull * nBitsPerComponent * verts.size();
        out.bitstream.assign((size_t)((totalBits + 7) / 8), 0);

        uint64_t bitOff = 0;
        for (size_t i = 0; i < verts.size(); ++i)
        {
            uint32_t ox = (uint32_t)(qX[i] - aX);  if (ox > maxOff) ox = maxOff;
            uint32_t oy = (uint32_t)(qY[i] - aY);  if (oy > maxOff) oy = maxOff;
            uint32_t oz = (uint32_t)(qZ[i] - aZ);  if (oz > maxOff) oz = maxOff;
            WriteBits(out.bitstream, bitOff, ox, nBitsPerComponent); bitOff += nBitsPerComponent;
            WriteBits(out.bitstream, bitOff, oy, nBitsPerComponent); bitOff += nBitsPerComponent;
            WriteBits(out.bitstream, bitOff, oz, nBitsPerComponent); bitOff += nBitsPerComponent;
        }

        return out;
    }

    // CPU-side reference decoder used to verify roundtrip correctness during init.
    inline std::vector<ProceduralGeometry::float3> Decode(const EncodedCluster& enc)
    {
        const int e  = (int)DECODE_D3D12_COMPRESSED1_EXPONENT(enc.header);
        const int aX = (int)DECODE_D3D12_COMPRESSED1_X_ANCHOR(enc.header);
        const int aY = (int)DECODE_D3D12_COMPRESSED1_Y_ANCHOR(enc.header);
        const int aZ = (int)DECODE_D3D12_COMPRESSED1_Z_ANCHOR(enc.header);
        const int nx = (int)DECODE_D3D12_COMPRESSED1_X_BITS(enc.header) + 1;
        const int ny = (int)DECODE_D3D12_COMPRESSED1_Y_BITS(enc.header) + 1;
        const int nz = (int)DECODE_D3D12_COMPRESSED1_Z_BITS(enc.header) + 1;
        const float unit = std::ldexp(1.0f, e - 127);

        std::vector<ProceduralGeometry::float3> out(enc.vertexCount);
        uint64_t bitOff = 0;
        for (unsigned int i = 0; i < enc.vertexCount; ++i)
        {
            uint32_t ox = ReadBits(enc.bitstream, bitOff, nx); bitOff += nx;
            uint32_t oy = ReadBits(enc.bitstream, bitOff, ny); bitOff += ny;
            uint32_t oz = ReadBits(enc.bitstream, bitOff, nz); bitOff += nz;
            out[i].x = (float)(aX + (int32_t)ox) * unit;
            out[i].y = (float)(aY + (int32_t)oy) * unit;
            out[i].z = (float)(aZ + (int32_t)oz) * unit;
        }
        return out;
    }
}
