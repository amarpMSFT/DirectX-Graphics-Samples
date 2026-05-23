//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// GpuBuffer.h
//
// Skeleton-grade GPU buffer helpers used by the milestone-2a path.  Two
// classes:
//
//   UploadBuffer  - upload-heap buffer for one-time CPU->GPU data uploads
//                   (vertex / index data, instance descriptors).  Map once,
//                   keep persistently mapped; CPU writes go straight through.
//
//   DefaultBuffer - default-heap buffer for GPU-only data (BLAS storage,
//                   PTLAS storage, scratch, RWStructuredBuffer outputs).
//                   Use UploadBuffer + copy commands if you need to seed it.
//
// Both use CreateCommittedResource.  Heap-backed sub-allocation comes later
// when the per-frame allocator (GpuMemory) lands -- for now each buffer is
// its own resource.  See docs/design.md for the eventual plan.
//
#pragma once

#include "DXSampleHelper.h"
#include <cstring>

namespace PtSample
{
    inline Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateUploadBuffer(ID3D12Device* device, UINT64 sizeBytes, const wchar_t* name = nullptr)
    {
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeBytes);
        Microsoft::WRL::ComPtr<ID3D12Resource> r;
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&r)));
        if (name) r->SetName(name);
        return r;
    }

    inline Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateDefaultBuffer(ID3D12Device* device, UINT64 sizeBytes,
                        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON,
                        const wchar_t* name = nullptr)
    {
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeBytes, flags);
        Microsoft::WRL::ComPtr<ID3D12Resource> r;
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE,
            &desc, initialState,
            nullptr, IID_PPV_ARGS(&r)));
        if (name) r->SetName(name);
        return r;
    }

    // Copies bytes into a newly-mapped upload buffer.  Buffer is left
    // mapped (upload heaps are CPU-coherent so no Unmap is required).
    inline Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateUploadBufferWithData(ID3D12Device* device,
                               const void* src, UINT64 sizeBytes,
                               const wchar_t* name = nullptr)
    {
        auto r = CreateUploadBuffer(device, sizeBytes, name);
        void* p = nullptr;
        CD3DX12_RANGE noRead(0, 0);
        ThrowIfFailed(r->Map(0, &noRead, &p));
        std::memcpy(p, src, (size_t)sizeBytes);
        // Leave mapped; release will unmap.
        return r;
    }
}
