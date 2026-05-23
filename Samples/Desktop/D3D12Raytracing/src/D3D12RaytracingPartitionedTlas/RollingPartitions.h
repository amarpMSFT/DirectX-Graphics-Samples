//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// RollingPartitions.h
//
// Manages a **rolling partition budget** that follows the flock through
// the static ball lattice.  Concept:
//
//   * Balls live at FIXED world positions (the lattice never moves).
//   * The lattice is logically divided into CELLS (one per grid cube) --
//     same cell layout the phase-2 grid used.
//   * A PTLAS has a fixed budget of `P` partitions, where `P` is typically
//     SMALLER than total cell count (controlled via --partitions and, in
//     phase 3c, a runtime UI knob that triggers PTLAS resize).
//   * Each frame, the `P` cells CLOSEST TO THE FLOCK (with a small forward
//     bias) are picked as the active set.  Each active cell holds one PTLAS
//     partition slot; balls in that cell are visible.  Cells outside the
//     active set: their balls are DISABLED (PTLAS instance with
//     AccelerationStructure = NULL).
//   * As the flock flies, the active set shifts.  Partition slots are
//     RECYCLED -- a slot that was assigned to a cell behind the flock gets
//     reassigned to a fresh cell ahead, with balls re-bound via WRITE_INSTANCE.
//
// Exactly matches the user's description: "new partitions are formed in
// the direction of travel in the distance by moving invisible ones far
// behind up forward."
//
// Update() is O(cells * log cells) per frame for the sort, plus
// O(cells + balls-in-changed-cells) for the assignment.  216 cells + a few
// thousand balls is trivial; large worlds want a more incremental approach
// (a future refinement).
//
#pragma once

#include "SceneLayout.h"

#include <DirectXMath.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>

class RollingPartitions
{
public:
    static constexpr uint32_t kNoPartition = 0xFFFFFFFFu;

    // Set up once at init.  `partitionBudget` = the number of PTLAS
    // partition slots available; `forwardBias` is a unitless preference
    // weight for cells in the flock's forward direction (0 = pure distance,
    // 0.5 = strongly prefer ahead).  `scene` and `ballPositions` provide
    // the cell layout + ball-to-cell assignment.
    void Initialize(uint32_t partitionBudget,
                    float forwardBias,
                    const SceneLayout& scene,
                    const std::vector<DirectX::XMFLOAT3>& ballPositions)
    {
        m_budget       = partitionBudget;
        m_forwardBias  = forwardBias;
        m_ballPos      = ballPositions;
        m_ballOwner.assign(ballPositions.size(), kNoPartition);
        m_partitionCell.assign(m_budget, kNoPartition);
        m_cellPartition.assign(scene.Partitions(), kNoPartition);
        m_partitionHome.assign(m_budget, DirectX::XMFLOAT3{ 0, 0, 0 });
        m_cellCentroid.assign(scene.Partitions(), DirectX::XMFLOAT3{ 0, 0, 0 });
        m_cellBalls.assign(scene.Partitions(), {});

        // Cell centroids and ball -> cell membership.  Cell index = the
        // SceneLayout grid cell index (the original "partition" from
        // phase 2b/c, now reinterpreted as just a spatial cell with no
        // direct PTLAS-partition identity).
        m_scene = scene;
        for (uint32_t pk = 0; pk < scene.gridZ; ++pk)
        for (uint32_t pj = 0; pj < scene.gridY; ++pj)
        for (uint32_t pi = 0; pi < scene.gridX; ++pi)
        {
            uint32_t cellIdx = scene.PartitionIndex(pi, pj, pk);
            m_cellCentroid[cellIdx] = scene.PartitionHomeWorld(pi, pj, pk);
            for (uint32_t bk = 0; bk < scene.ballsPerSide; ++bk)
            for (uint32_t bj = 0; bj < scene.ballsPerSide; ++bj)
            for (uint32_t bi = 0; bi < scene.ballsPerSide; ++bi)
            {
                uint32_t b = scene.BallIndex(pi, pj, pk, bi, bj, bk);
                m_cellBalls[cellIdx].push_back(b);
            }
        }
    }

    uint32_t PartitionCount() const { return m_budget; }
    uint32_t BallCount()      const { return (uint32_t)m_ballPos.size(); }
    uint32_t Owner(uint32_t ballIdx) const { return m_ballOwner[ballIdx]; }
    const DirectX::XMFLOAT3& BallPos(uint32_t ballIdx) const { return m_ballPos[ballIdx]; }
    const DirectX::XMFLOAT3& PartitionHome(uint32_t p) const { return m_partitionHome[p]; }
    bool PartitionActive(uint32_t p) const { return m_partitionCell[p] != kNoPartition; }

    // Balls whose owner changed this frame (subset of [0, BallCount())).
    const std::vector<uint32_t>& ChangedBalls() const { return m_changedBalls; }

    // Recompute the rolling window for one frame.  `flockForward` should
    // be approximately unit length.  Fills internal state; callers query
    // via ChangedBalls()/Owner()/PartitionHome().
    void Update(const DirectX::XMFLOAT3& flockPos, const DirectX::XMFLOAT3& flockForward)
    {
        const uint32_t cellCount = (uint32_t)m_cellCentroid.size();
        const uint32_t budget    = std::min(m_budget, cellCount);

        // 1) Score every cell: distance to flock, with a soft preference
        //    for cells ahead of the flock along forward.  Lower score = pick first.
        struct Scored { float score; uint32_t cell; };
        std::vector<Scored> scored(cellCount);
        for (uint32_t c = 0; c < cellCount; ++c)
        {
            const auto& ctr = m_cellCentroid[c];
            float dx = ctr.x - flockPos.x;
            float dy = ctr.y - flockPos.y;
            float dz = ctr.z - flockPos.z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            float fwd  = dx*flockForward.x + dy*flockForward.y + dz*flockForward.z;
            scored[c] = { dist - m_forwardBias * fwd, c };
        }
        // Partial-sort the smallest `budget` elements to the front.
        std::nth_element(scored.begin(),
                         scored.begin() + budget,
                         scored.end(),
                         [](const Scored& a, const Scored& b){ return a.score < b.score; });

        // 2) Build the new active set + mark cells.
        std::vector<bool> nowActive(cellCount, false);
        for (uint32_t i = 0; i < budget; ++i)
            nowActive[scored[i].cell] = true;

        // 3) Reuse existing partition->cell assignments where the cell is
        //    still active.  Mark surviving partitions; freed-up partitions
        //    go into the pool.
        std::vector<uint32_t> freedPartitions;
        freedPartitions.reserve(m_budget);
        for (uint32_t p = 0; p < m_budget; ++p)
        {
            uint32_t oldCell = m_partitionCell[p];
            if (oldCell != kNoPartition && nowActive[oldCell])
            {
                // Keep partition->cell binding (m_cellPartition will be
                // refilled below).
            }
            else
            {
                if (oldCell != kNoPartition)
                {
                    m_cellPartition[oldCell] = kNoPartition;   // old cell loses its partition
                    m_partitionCell[p]       = kNoPartition;
                }
                freedPartitions.push_back(p);
            }
        }

        // 4) Re-fill m_cellPartition from surviving bindings.
        for (uint32_t p = 0; p < m_budget; ++p)
        {
            uint32_t cell = m_partitionCell[p];
            if (cell != kNoPartition) m_cellPartition[cell] = p;
        }

        // 5) Bind freed partitions to newly-active cells that don't have one.
        uint32_t fpIdx = 0;
        for (uint32_t i = 0; i < budget; ++i)
        {
            uint32_t cell = scored[i].cell;
            if (m_cellPartition[cell] != kNoPartition) continue; // already kept
            uint32_t p = freedPartitions[fpIdx++];
            m_cellPartition[cell] = p;
            m_partitionCell[p]    = cell;
            m_partitionHome[p]    = m_cellCentroid[cell];
        }
        // Update partition homes for kept bindings too (the home shouldn't
        // change since the cell didn't move, but be safe).
        for (uint32_t p = 0; p < m_budget; ++p)
        {
            uint32_t cell = m_partitionCell[p];
            if (cell != kNoPartition) m_partitionHome[p] = m_cellCentroid[cell];
        }

        // 6) Compute per-ball owner changes.
        m_changedBalls.clear();
        for (uint32_t c = 0; c < cellCount; ++c)
        {
            uint32_t newOwner = nowActive[c] ? m_cellPartition[c] : kNoPartition;
            for (uint32_t b : m_cellBalls[c])
            {
                if (m_ballOwner[b] != newOwner)
                {
                    m_ballOwner[b] = newOwner;
                    m_changedBalls.push_back(b);
                }
            }
        }
    }

private:
    SceneLayout m_scene = {};
    uint32_t m_budget       = 0;
    float    m_forwardBias  = 0.3f;

    std::vector<DirectX::XMFLOAT3> m_ballPos;       // fixed world positions
    std::vector<uint32_t>          m_ballOwner;     // current partition idx or kNoPartition

    std::vector<DirectX::XMFLOAT3> m_cellCentroid;  // per fixed cell, world centroid
    std::vector<std::vector<uint32_t>> m_cellBalls; // per fixed cell, list of ball indices
    std::vector<uint32_t>          m_cellPartition; // per cell, currently-assigned partition idx (or kNoPartition)

    std::vector<uint32_t>          m_partitionCell; // per PTLAS partition, currently-assigned cell idx (or kNoPartition)
    std::vector<DirectX::XMFLOAT3> m_partitionHome; // per PTLAS partition, current world centroid

    std::vector<uint32_t>          m_changedBalls;  // balls whose owner changed THIS frame
};
