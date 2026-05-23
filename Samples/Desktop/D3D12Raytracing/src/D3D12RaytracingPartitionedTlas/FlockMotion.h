//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// FlockMotion.h
//
// Smooth flock-center motion + a camera-follow rig.  Header-only.
//
// The flock travels along a multi-frequency Lissajous-style curve that
// gradually arcs through 3D, with no obvious looping period for the
// observer.  The camera trails behind the flock by `trailDistance` units
// (along the negative-velocity direction) and slightly above (`trailHeight`).
// It looks ahead of the flock by `lookAheadDistance`.
//
// All parameters live in the FlockMotion struct so tweaks are one-stop.
//
#pragma once

#include <DirectXMath.h>
#include <cmath>

struct FlockMotion
{
    // Path shape.  pathRadius scales the orbit envelope; freq* control how
    // each axis sweeps.  Pick irrational ratios so the curve doesn't
    // visibly repeat.
    float pathRadius   = 5.5f;       // wraps within the ball lattice
    float freqX        = 0.085f;     // rad/s
    float freqY        = 0.043f;
    float freqZ        = 0.061f;
    float phaseY       = 1.3f;
    float phaseZ       = 0.4f;
    float vertScale    = 0.55f;      // y motion is gentler than xz

    // Camera follow rig.
    float trailDistance   = 2.8f;   // how far behind the flock the camera sits
    float trailHeight     = 0.7f;   // how far above the flock
    float lookAheadDist   = 2.0f;   // where the camera looks (ahead of flock)
    float minVelLen       = 0.05f;  // floor for velocity direction (avoids snap)

    // Computed each frame:
    DirectX::XMFLOAT3 position   = { 0, 0, 0 };
    DirectX::XMFLOAT3 velocity   = { 1, 0, 0 };
    DirectX::XMFLOAT3 forward    = { 1, 0, 0 };  // smoothed velocity direction

    // Update for a given wall-clock time.  `smoothing` is the per-call
    // weight for blending new forward direction toward instantaneous
    // velocity direction (1.0 = snap, 0.0 = never update).
    void Update(double tsec, float smoothing = 0.10f)
    {
        const float t = (float)tsec;
        // Path.
        position = {
            pathRadius * std::sin(t * freqX),
            vertScale * pathRadius * std::sin(t * freqY + phaseY),
            pathRadius * std::sin(t * freqZ + phaseZ),
        };
        // Derivative gives an instantaneous velocity direction.
        velocity = {
            pathRadius * freqX * std::cos(t * freqX),
            vertScale * pathRadius * freqY * std::cos(t * freqY + phaseY),
            pathRadius * freqZ * std::cos(t * freqZ + phaseZ),
        };
        // Smooth the forward direction so the camera doesn't snap when
        // velocity crosses zero (at path extremes).
        float vlen = std::sqrt(velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z);
        if (vlen > minVelLen)
        {
            DirectX::XMFLOAT3 vdir = { velocity.x / vlen, velocity.y / vlen, velocity.z / vlen };
            forward.x = forward.x + smoothing * (vdir.x - forward.x);
            forward.y = forward.y + smoothing * (vdir.y - forward.y);
            forward.z = forward.z + smoothing * (vdir.z - forward.z);
            // Re-normalise to keep forward unit-length.
            float flen = std::sqrt(forward.x*forward.x + forward.y*forward.y + forward.z*forward.z);
            if (flen > 1e-6f) { forward.x/=flen; forward.y/=flen; forward.z/=flen; }
        }
    }

    // Camera position trailing the flock; looks ahead of flock.
    DirectX::XMFLOAT3 CameraPos() const
    {
        return {
            position.x - trailDistance * forward.x,
            position.y - trailDistance * forward.y + trailHeight,
            position.z - trailDistance * forward.z,
        };
    }
    DirectX::XMFLOAT3 CameraTarget() const
    {
        return {
            position.x + lookAheadDist * forward.x,
            position.y + lookAheadDist * forward.y,
            position.z + lookAheadDist * forward.z,
        };
    }
};
