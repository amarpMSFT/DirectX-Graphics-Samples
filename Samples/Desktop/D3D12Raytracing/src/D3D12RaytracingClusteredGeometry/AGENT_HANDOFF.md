# DXR2 cross-machine handoff

The detailed handoff and v0.29 runtime compatibility patch are maintained in the internal Direct3D repository branch:

- Repository: `Direct3D`
- Branch: `user/amarp/dxr2-v029-aabb-compat`
- Handoff commit: `bb4081114`
- Runtime compatibility commit: `8d2867730`
- File: `DXR2_AGENT_HANDOFF.md`
- Bundle: `DXR2AgentHandoff/`

Paths differ by machine. Discover the local Direct3D and DirectX-Graphics-Samples clone roots before following the internal handoff; do not assume this machine's `D:\...` paths.

The receiving work machine has an NVIDIA GeForce RTX 5070. Establish and record its exact driver version, adapter index, and DXR2 capability results before testing. Use the same workflow documented in the internal handoff, but do not compare RTX 5070 performance numbers directly against the source machine's RTX 4090 numbers as a correctness criterion.

Before returning work to another agent, update the internal `DXR2_AGENT_HANDOFF.md`, refresh its evidence/patch manifest, preserve CRLF, and restore `d3dconfig device force-warp=false`.

Sample source state:

- Current implementation commit: `3b1e0f5e`
- v0.29 sample compatibility branch: `user/amarp/dxr2-v029-sample`
- v0.29 sample compatibility commit: `6cc497ef`
- Never push this repository to `origin`; use the configured fork remote only when authorized.
