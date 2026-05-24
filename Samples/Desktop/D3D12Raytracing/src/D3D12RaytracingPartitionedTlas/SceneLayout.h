//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// SceneLayout.h
//
// Static scene layout: a 3D grid of partitions, each holding an NxNxN sub-
// grid of balls.  Balls are equally spaced across the entire global lattice
// (including across partition boundaries) so the partitioning is purely a
// logical / accel-structure concept, not a visual one.
//
// Index conventions:
//   * Partition (pi, pj, pk) with pi in [0, gridX), etc.
//     -> linear partition index = pi + pj*gridX + pk*gridX*gridY
//   * Ball within a partition (bi, bj, bk) with bi in [0, ballsPerSide), etc.
//   * Global ball index linearises ALL balls across all partitions:
//       linear = ((pk*gridY + pj)*gridX + pi) * ballsPerPart
//                + ((bk*N + bj)*N + bi)
//     where N = ballsPerSide, ballsPerPart = N*N*N.
//
#pragma once

#include <DirectXMath.h>
#include <cstdint>

struct SceneLayout
{
    // Grid sizing.  Phase 2b stress-test defaults: 216 partitions, 5832
    // ball instances.  Each partition contains 27 balls (3 per side).
    // The 24×25×25 / NxNxN target from the design lands in phase 3+; this
    // value is the largest scene that still draws in a small fraction of
    // a second on warm cache, so iteration stays fast.
    uint32_t gridX        = 6;
    uint32_t gridY        = 6;
    uint32_t gridZ        = 6;
    uint32_t ballsPerSide = 3;   // N in each axis per partition
    float    ballSpacing  = 1.5f;
    float    ballScale    = 0.30f;  // radius (unit-sphere scaled) -- smaller per user request

    // Derived counts.
    uint32_t Partitions()       const { return gridX * gridY * gridZ; }
    uint32_t BallsPerPartition() const { return ballsPerSide * ballsPerSide * ballsPerSide; }
    uint32_t TotalBalls()        const { return Partitions() * BallsPerPartition(); }

    // Region (partition) extents in world units.
    DirectX::XMFLOAT3 RegionSize() const
    {
        return { ballSpacing * ballsPerSide,
                 ballSpacing * ballsPerSide,
                 ballSpacing * ballsPerSide };
    }

    // Total scene size in world units.
    DirectX::XMFLOAT3 SceneSize() const
    {
        return { ballSpacing * ballsPerSide * gridX,
                 ballSpacing * ballsPerSide * gridY,
                 ballSpacing * ballsPerSide * gridZ };
    }

    // Center the world about origin so the orbit camera can frame it
    // without per-frame fuss.
    DirectX::XMFLOAT3 SceneCenterToOriginShift() const
    {
        auto s = SceneSize();
        return { -0.5f * s.x + 0.5f * ballSpacing,
                 -0.5f * s.y + 0.5f * ballSpacing,
                 -0.5f * s.z + 0.5f * ballSpacing };
    }

    // World-space center of the partition cell (centroid of its balls'
    // positions).  Used as the "partition home" for the camera-anchored
    // partition-translation scheme: each frame we set partition translation
    // = PartitionHome - as_origin, so the PTLAS sees positions near zero.
    DirectX::XMFLOAT3 PartitionHomeWorld(uint32_t pi, uint32_t pj, uint32_t pk) const
    {
        // Centroid of the NxNxN ball positions in this partition.  Since
        // balls are at (partI*N + 0..N-1) * spacing + shift, the centroid
        // along each axis is partI*N + (N-1)/2 * spacing + shift, which
        // simplifies to BallWorldPos(pi,pj,pk, (N-1)/2,(N-1)/2,(N-1)/2)
        // for the closest integer mid-ball (good enough; for even N we
        // shift by an extra 0.5 cell, also fine).
        auto shift = SceneCenterToOriginShift();
        const float halfN = 0.5f * (float)(ballsPerSide - 1);
        return {
            shift.x + ballSpacing * (pi * ballsPerSide + halfN),
            shift.y + ballSpacing * (pj * ballsPerSide + halfN),
            shift.z + ballSpacing * (pk * ballsPerSide + halfN),
        };
    }

    DirectX::XMFLOAT3 PartitionHomeWorld(uint32_t linearIdx) const
    {
        uint32_t pk = linearIdx / (gridX * gridY);
        uint32_t pj = (linearIdx / gridX) % gridY;
        uint32_t pi =  linearIdx % gridX;
        return PartitionHomeWorld(pi, pj, pk);
    }

    // ---- Helpers ----

    uint32_t PartitionIndex(uint32_t pi, uint32_t pj, uint32_t pk) const
    {
        return pi + pj * gridX + pk * gridX * gridY;
    }

    // Linear ball index across the whole scene.
    uint32_t BallIndex(uint32_t pi, uint32_t pj, uint32_t pk,
                       uint32_t bi, uint32_t bj, uint32_t bk) const
    {
        const uint32_t N = ballsPerSide;
        return PartitionIndex(pi, pj, pk) * BallsPerPartition()
             + ((bk * N + bj) * N + bi);
    }

    // World-space position for one ball.
    DirectX::XMFLOAT3 BallWorldPos(uint32_t pi, uint32_t pj, uint32_t pk,
                                   uint32_t bi, uint32_t bj, uint32_t bk) const
    {
        auto shift = SceneCenterToOriginShift();
        return {
            shift.x + ballSpacing * (pi * ballsPerSide + bi),
            shift.y + ballSpacing * (pj * ballsPerSide + bj),
            shift.z + ballSpacing * (pk * ballsPerSide + bk),
        };
    }
};
