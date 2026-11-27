// Our own test shader: a 4x4-thread group fill of an R32_UINT texture through a UAV table; the
// texel at (x, y) is k + x + 4 * y (k a root constant).
cbuffer C : register(b0) { uint k; };
RWTexture2D<uint> dst : register(u0);
[RootSignature("RootConstants(num32BitConstants=1, b0), DescriptorTable(UAV(u0))")]
[numthreads(4, 4, 1)]
void main(uint3 id : SV_DispatchThreadID) { dst[id.xy] = k + id.x + 4 * id.y; }
