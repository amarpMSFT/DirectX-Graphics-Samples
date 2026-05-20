// RaytracingHlslCompat.h -- shared types between C++ and HLSL.

#ifdef HLSL
#include "HlslCompat.h"
#else
using namespace DirectX;
typedef DirectX::XMFLOAT2   XMFLOAT2;
typedef DirectX::XMFLOAT3   XMFLOAT3;
typedef DirectX::XMFLOAT4   XMFLOAT4;
typedef DirectX::XMUINT2    XMUINT2;
typedef DirectX::XMUINT3    XMUINT3;
typedef DirectX::XMUINT4    XMUINT4;
typedef DirectX::XMMATRIX   XMMATRIX;
typedef UINT                uint;
#endif

struct SceneConstantBuffer
{
    XMMATRIX  viewToWorld;       // ray gen camera basis
    XMFLOAT4  cameraPosition;    // .xyz = world-space eye position
    XMFLOAT4  miscParams;        // .x = aspect ratio, .y = tan(fov/2)
};
