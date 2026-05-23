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

// =============================================================================
// PartitionedTlasSample (MILESTONE 1 SKELETON)
//
// What works in this file today:
//   * D3D12 device + DXR2 surfaces (queried; missing-feature is logged-only,
//     so the sample still launches on machines without the experimental SDK)
//   * Clears the back buffer to a distinctive teal so a screenshot
//     unambiguously identifies THIS sample (vs. clustered etc.)
//   * --screenshot N path / --exit-after-frames N CLI for headless capture
//     (mirrors the clustered-geometry sample's wrapper-script contract)
//
// What is intentionally NOT here yet (see docs/design.md for the roadmap):
//   * PtlasSystem / ClusterSystem / SceneState / GpuMemory
//   * Raytracing pipeline / shader tables
//   * Any actual PTLAS operations
//
// =============================================================================

#include "stdafx.h"
#include "PartitionedTlasSample.h"

using Microsoft::WRL::ComPtr;

PartitionedTlasSample::PartitionedTlasSample(UINT width, UINT height, std::wstring name) :
    DXSample(width, height, name)
{
}

// ---------------------------------------------------------------------------
// CLI args.  Conservative subset of the clustered-geometry sample's CLI for
// now -- enough to drive headless screenshot capture and clean-exit.  Extend
// as PTLAS features land.
// ---------------------------------------------------------------------------
void PartitionedTlasSample::ParseCommandLineArgs(_In_reads_(argc) WCHAR* argv[], int argc)
{
    DXSample::ParseCommandLineArgs(argv, argc);
    for (int i = 1; i < argc; i++)
    {
        if (_wcsicmp(argv[i], L"--screenshot") == 0 && i + 2 < argc)
        {
            m_screenshotFrame = _wtoi(argv[i + 1]);
            m_screenshotPath  = argv[i + 2];
            i += 2;
        }
        else if (_wcsicmp(argv[i], L"--exit-after-frames") == 0 && i + 1 < argc)
        {
            int n = _wtoi(argv[i + 1]);
            m_exitAfterFrames = (UINT)std::max(0, n);
            i += 1;
        }
    }
    SampleLog::LogF(L"OnInit: CLI parsed (screenshotFrame=%d, exitAfterFrames=%u)\n",
                    m_screenshotFrame, m_exitAfterFrames);
}

// ---------------------------------------------------------------------------
// OnInit: enable DXR2 experimental features, create device + swap chain,
// query Clusters+PTLAS support.  Modelled on the clustered-geometry sample's
// OnInit but stripped down -- no scene, no AS, no shaders yet.
// ---------------------------------------------------------------------------
void PartitionedTlasSample::OnInit()
{
    // DRED for diagnostics if the GPU later hangs on PTLAS work.  Costs ~0;
    // captures GPU breadcrumbs + page-fault info into the log on TDR/removal.
    {
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dredSettings;
        HRESULT hrDred = D3D12GetDebugInterface(IID_PPV_ARGS(&dredSettings));
        if (SUCCEEDED(hrDred) && dredSettings) {
            dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            SampleLog::Write(L"OnInit: DRED enabled (breadcrumbs + page-fault)\n");
        } else {
            SampleLog::LogF(L"OnInit: DRED unavailable hr=0x%08X\n", (unsigned)hrDred);
        }
    }

    // DXR2 experimental-features opt-in.  Required BEFORE any D3D12CreateDevice
    // call in this process or ClustersAndPTLAS prebuild queries silently
    // return zero sizes even when the cap reports YES.
    {
        UUID experimentalFeatures[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
        HRESULT hrExp = D3D12EnableExperimentalFeatures(_countof(experimentalFeatures),
                                                        experimentalFeatures, nullptr, nullptr);
        SampleLog::LogF(L"OnInit: D3D12EnableExperimentalFeatures -> hr=0x%08X (%s)\n",
                        (unsigned)hrExp, SUCCEEDED(hrExp) ? L"OK" : L"FAIL");
        // Don't throw if it fails -- we still want the window to open + show
        // teal so the wrapper script can confirm the build/launch path works.
        // Subsequent DXR2 query will report unsupported and the sample
        // (when phase 2+ lands) will refuse to do PTLAS work but keep
        // rendering the placeholder clear.
    }

    SampleLog::Write(L"OnInit: creating DeviceResources\n");
    m_deviceResources = std::make_unique<DX::DeviceResources>(
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_UNKNOWN,
        FrameCount,
        D3D_FEATURE_LEVEL_11_0,
        /*options*/0,
        m_adapterIDoverride);
    m_deviceResources->RegisterDeviceNotify(this);
    m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
    m_deviceResources->InitializeDXGIAdapter();

    // Adapter sanity check + DXR1 floor.  PTLAS needs DXR2; we'll flag-only
    // here (not throw) so the window still opens on machines without DXR2.
    {
        ComPtr<ID3D12Device> testDevice;
        HRESULT hrCreate = D3D12CreateDevice(m_deviceResources->GetAdapter(),
                                             D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&testDevice));
        SampleLog::LogF(L"  D3D12CreateDevice -> hr=0x%08X (%s)\n",
                        (unsigned)hrCreate, SUCCEEDED(hrCreate) ? L"OK" : L"FAIL");
        ThrowIfFailed(hrCreate);
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
        HRESULT hrFeat = testDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5));
        SampleLog::LogF(L"  CheckFeatureSupport(OPTIONS5) hr=0x%08X, RaytracingTier=0x%X\n",
                        (unsigned)hrFeat, (unsigned)opts5.RaytracingTier);
        // Don't throw: we want the skeleton to launch even on DXR0 adapters.
    }

    m_deviceResources->CreateDeviceResources();
    m_deviceResources->CreateWindowSizeDependentResources();
    CreateDeviceDependentResources();
    SampleLog::Write(L"OnInit: complete\n");
}

void PartitionedTlasSample::CreateDeviceDependentResources()
{
    auto device      = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

    // QI for DXR1 + DXR2 surfaces.  Log-and-continue instead of throw -- a
    // DXR0 adapter should still get a teal clear via the swap-chain.
    HRESULT hr;
    hr = device     ->QueryInterface(IID_PPV_ARGS(&m_dxrDevice));
    SampleLog::LogF(L"  QI ID3D12Device5                 hr=0x%08X\n", (unsigned)hr);
    hr = commandList->QueryInterface(IID_PPV_ARGS(&m_dxrCommandList));
    SampleLog::LogF(L"  QI ID3D12GraphicsCommandList4    hr=0x%08X\n", (unsigned)hr);
    hr = device     ->QueryInterface(IID_PPV_ARGS(&m_dxr2Device));
    SampleLog::LogF(L"  QI ID3D12DeviceRaytracing2       hr=0x%08X%s\n", (unsigned)hr,
                    SUCCEEDED(hr) ? L"" : L"  (no DXR2 -- experimental D3D12Core not loaded?)");
    hr = commandList->QueryInterface(IID_PPV_ARGS(&m_dxr2CommandList));
    SampleLog::LogF(L"  QI ID3D12CommandListRaytracing2  hr=0x%08X\n", (unsigned)hr);

    // The DXR2 feature options enum name is provisional in the experimental
    // headers.  In the clustered sample it is queried via
    //   D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL .ClustersAndPTLASSupported
    // (the exact enum number changes with experimental SDK drops).  Try the
    // documented current name; tolerate it being missing.  Once the API is
    // finalized, switch to the official name.
    if (m_dxr2Device)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL optsExp = {};
        HRESULT hrCaps = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL,
                                                     &optsExp, sizeof(optsExp));
        if (SUCCEEDED(hrCaps))
        {
            m_clustersAndPtlasSupported = optsExp.ClustersAndPTLASSupported != FALSE;
            SampleLog::LogF(L"  CheckFeatureSupport(OPTIONS_EXPERIMENTAL) hr=0x%08X "
                            L"ClustersAndPTLASSupported=%s\n",
                            (unsigned)hrCaps, m_clustersAndPtlasSupported ? L"YES" : L"NO");
        }
        else
        {
            SampleLog::LogF(L"  CheckFeatureSupport(OPTIONS_EXPERIMENTAL) hr=0x%08X -- "
                            L"enum name probably renamed in this SDK; cap left as NO\n",
                            (unsigned)hrCaps);
        }
    }
    else
    {
        SampleLog::Write(L"  DXR2 unavailable -- PTLAS path will be inactive (clear only)\n");
    }

    // Adapter name for the log (helpful when WARP vs HW swap is in play).
    auto adapterDesc = m_deviceResources->GetAdapterDescription();
    SampleLog::LogF(L"  Adapter: %s\n", adapterDesc);
}

void PartitionedTlasSample::ReleaseDeviceDependentResources()
{
    m_dxr2CommandList.Reset();
    m_dxr2Device     .Reset();
    m_dxrCommandList .Reset();
    m_dxrDevice      .Reset();
    m_clustersAndPtlasSupported = false;
}

void PartitionedTlasSample::OnUpdate()
{
    m_timer.Tick();
}

void PartitionedTlasSample::OnRender()
{
    if (!m_deviceResources->IsWindowVisible()) return;

    m_deviceResources->Prepare();
    auto commandList = m_deviceResources->GetCommandList();
    auto rtv         = m_deviceResources->GetRenderTargetView();

    // Teal so a screenshot of THIS sample is visually distinct from any other
    // sample that defaults to black/grey.  R=0, G=0.5, B=0.55.
    const float kClearColor[4] = { 0.0f, 0.50f, 0.55f, 1.0f };
    commandList->ClearRenderTargetView(rtv, kClearColor, 0, nullptr);

    m_deviceResources->Present();
    ++m_framesRendered;

    // Headless capture: on the requested frame, screenshot once and (if
    // --exit-after-frames wasn't also given) post-quit so wrapper scripts get
    // a clean exit.  The capture path uses the PREVIOUS-frame back buffer
    // index because we already Present()ed above, so the just-cleared image
    // is what was sent to display.
    if (m_screenshotFrame >= 0 && (UINT)m_screenshotFrame < m_framesRendered && !m_screenshotTaken)
    {
        m_screenshotTaken = true;
        SampleLog::LogF(L"[screenshot] capturing frame %u to %s\n",
                        m_framesRendered, m_screenshotPath.c_str());
        CaptureBackBufferToFile(m_screenshotPath);
        if (m_exitAfterFrames == 0)
        {
            PostQuitMessage(0);
            return;
        }
    }

    if (m_exitAfterFrames > 0 && m_framesRendered >= m_exitAfterFrames)
    {
        PostQuitMessage(0);
    }
}

void PartitionedTlasSample::OnSizeChanged(UINT width, UINT height, bool minimized)
{
    if (!m_deviceResources->WindowSizeChanged(width, height, minimized))
    {
        return;
    }
    UpdateForSizeChange(width, height);
}

void PartitionedTlasSample::OnDestroy()
{
    m_deviceResources->WaitForGpu();
    OnDeviceLost();
}

void PartitionedTlasSample::OnDeviceLost()
{
    ReleaseDeviceDependentResources();
}

void PartitionedTlasSample::OnDeviceRestored()
{
    CreateDeviceDependentResources();
}

// =============================================================================
// Screenshot capture (lifted verbatim from D3D12RaytracingClusteredGeometry's
// CaptureBackBufferToFile / SaveBGRAToPng -- same swap-chain semantics, same
// WIC pipeline).  Kept self-contained so this skeleton has zero NuGet deps.
// =============================================================================
void PartitionedTlasSample::CaptureBackBufferToFile(const std::wstring& path)
{
    auto device       = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();

    UINT capturedBackBufferIndex = m_deviceResources->GetPreviousFrameIndex();
    auto swapChain = m_deviceResources->GetSwapChain();
    ComPtr<ID3D12Resource> backBuffer;
    ThrowIfFailed(swapChain->GetBuffer(capturedBackBufferIndex, IID_PPV_ARGS(&backBuffer)));

    auto rtDesc = backBuffer->GetDesc();
    const UINT width  = (UINT)rtDesc.Width;
    const UINT height = (UINT)rtDesc.Height;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT64 totalBytes = 0; UINT rowCount = 0; UINT64 rowSizeInBytes = 0;
    device->GetCopyableFootprints(&rtDesc, 0, 1, 0, &footprint, &rowCount, &rowSizeInBytes, &totalBytes);

    auto heapPropsRB = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bufDesc     = CD3DX12_RESOURCE_DESC::Buffer(totalBytes);
    ComPtr<ID3D12Resource> readback;
    ThrowIfFailed(device->CreateCommittedResource(&heapPropsRB, D3D12_HEAP_FLAG_NONE,
        &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)));

    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)));
    {
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(),
            D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &toCopy);
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = backBuffer.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
        cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        auto toPresent = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &toPresent);
    }
    ThrowIfFailed(cl->Close());
    ID3D12CommandList* cls[] = { cl.Get() };
    commandQueue->ExecuteCommandLists(1, cls);
    m_deviceResources->WaitForGpu();

    void* mappedRaw = nullptr;
    D3D12_RANGE readRange = { 0, (SIZE_T)totalBytes };
    ThrowIfFailed(readback->Map(0, &readRange, &mappedRaw));
    HRESULT hrSave = SaveBGRAToPng(path, width, height,
        (const uint8_t*)mappedRaw + footprint.Offset, footprint.Footprint.RowPitch);
    D3D12_RANGE writeRange = { 0, 0 };
    readback->Unmap(0, &writeRange);

    SampleLog::LogF(L"[screenshot] %s %ux%u -> %s\n",
        SUCCEEDED(hrSave) ? L"wrote" : L"FAILED to write", width, height, path.c_str());
}

HRESULT PartitionedTlasSample::SaveBGRAToPng(const std::wstring& path,
                                             UINT width, UINT height,
                                             const uint8_t* data,
                                             UINT rowPitchBytes)
{
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool initedCom = SUCCEEDED(hr);
    auto cleanup = [&](HRESULT result) -> HRESULT { if (initedCom) CoUninitialize(); return result; };
    ComPtr<IWICImagingFactory> factory;
    if (FAILED(hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) return cleanup(hr);
    ComPtr<IWICStream> stream;
    if (FAILED(hr = factory->CreateStream(&stream))) return cleanup(hr);
    if (FAILED(hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE))) return cleanup(hr);
    ComPtr<IWICBitmapEncoder> encoder;
    if (FAILED(hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder))) return cleanup(hr);
    if (FAILED(hr = encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return cleanup(hr);
    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> props;
    if (FAILED(hr = encoder->CreateNewFrame(&frame, &props))) return cleanup(hr);
    if (FAILED(hr = frame->Initialize(props.Get()))) return cleanup(hr);
    if (FAILED(hr = frame->SetSize(width, height))) return cleanup(hr);
    WICPixelFormatGUID pf = GUID_WICPixelFormat32bppBGRA;
    if (FAILED(hr = frame->SetPixelFormat(&pf))) return cleanup(hr);
    const UINT tightStride = width * 4;
    if (rowPitchBytes == tightStride)
    {
        if (FAILED(hr = frame->WritePixels(height, tightStride, tightStride * height,
                                           const_cast<BYTE*>(data)))) return cleanup(hr);
    }
    else
    {
        std::vector<uint8_t> packed((size_t)tightStride * height);
        for (UINT y = 0; y < height; ++y)
            memcpy(packed.data() + (size_t)y * tightStride,
                   data + (size_t)y * rowPitchBytes, tightStride);
        if (FAILED(hr = frame->WritePixels(height, tightStride, tightStride * height,
                                           packed.data()))) return cleanup(hr);
    }
    if (FAILED(hr = frame->Commit()))   return cleanup(hr);
    if (FAILED(hr = encoder->Commit())) return cleanup(hr);
    return cleanup(S_OK);
}
