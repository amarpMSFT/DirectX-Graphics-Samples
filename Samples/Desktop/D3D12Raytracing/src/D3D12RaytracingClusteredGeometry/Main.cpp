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
#include "D3D12RaytracingClusteredGeometry.h"

// Agility SDK loader hooks. The system d3d12.dll uses these exports to find the
// matching D3D12Core.dll at runtime. The DLL is copied into bin\<cfg>\D3D12\
// by ExperimentalD3D12.props.
//
// The experimental D3D12Core in this project is a *preview* build, so we export
// D3D12_PREVIEW_SDK_VERSION (= 722 in d3d12.h at time of writing) rather than
// D3D12_SDK_VERSION (= 620, the latest non-preview release). When a real Agility
// SDK NuGet package shipping DXR2 ships, swap this back to D3D12_SDK_VERSION.
extern "C" { __declspec(dllexport) extern const UINT  D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath    = u8".\\D3D12\\"; }

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    SampleLog::Init();
    SampleLog::LogF(L"=== D3D12RaytracingClusteredGeometry starting (sdk=%u, preview=%u) ===\n",
                    (unsigned)D3D12_SDK_VERSION, (unsigned)D3D12_PREVIEW_SDK_VERSION);

    D3D12RaytracingClusteredGeometry sample(1280, 720, L"D3D12 Raytracing - Clustered Geometry");
    return Win32Application::Run(&sample, hInstance, nCmdShow);
}
