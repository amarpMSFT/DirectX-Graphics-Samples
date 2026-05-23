//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "stdafx.h"
#include "PartitionedTlasSample.h"

// Agility SDK loader hooks (matches the D3D12RaytracingClusteredGeometry sample).
// The system d3d12.dll uses these exports to find the matching D3D12Core.dll at
// runtime. The DLL is copied into bin\<cfg>\D3D12\ by ExperimentalD3D12.props.
//
// D3D12SDKVersion = 721 (release-SDK loader path, NOT the preview path):
// matches the experimental D3D12Core that supports DXR2 / PTLAS.
// When a real Agility SDK NuGet shipping DXR2 ships, swap this back to
// D3D12_SDK_VERSION.
extern "C" { __declspec(dllexport) extern const UINT  D3D12SDKVersion = 721; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath    = ".\\D3D12\\"; }

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    SampleLog::Init();
    SampleLog::LogF(L"=== D3D12RaytracingPartitionedTlas starting (sdk=%u, preview=%u) ===\n",
                    (unsigned)D3D12_SDK_VERSION, (unsigned)D3D12_PREVIEW_SDK_VERSION);

    PartitionedTlasSample sample(1280, 720, L"D3D12 Raytracing - Partitioned TLAS");
    return Win32Application::Run(&sample, hInstance, nCmdShow);
}
