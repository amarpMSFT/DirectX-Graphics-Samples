# DXR2 cross-machine handoff

The detailed living handoff, compatibility-runtime instructions, refreshed patches, and validation evidence are maintained in the internal Direct3D repository branch:

- Repository: `Direct3D`
- Branch: `user/amarp/dxr2-v029-aabb-compat`
- Current handoff commit: `9ecbb0364`
- Runtime compatibility implementation: `8d2867730`
- File: `DXR2_AGENT_HANDOFF.md`
- Bundle: `DXR2AgentHandoff/`

Paths differ by machine. Discover the local Direct3D and DirectX-Graphics-Samples clone roots before following the internal handoff; do not assume this machine's `D:\...` paths.

The receiving work machine has an NVIDIA GeForce RTX 5070. Establish and record its exact driver version, adapter index, and DXR2 capability results before testing. Use the same workflow documented in the internal handoff, but do not compare RTX 5070 performance numbers directly against the source RTX 4090 as a correctness criterion.

Before returning work to another agent, update internal `DXR2_AGENT_HANDOFF.md`, refresh its manifest/evidence, preserve CRLF, and restore `d3dconfig device force-warp=false`.

## Current sample state

- Current v0.30 branch: `user/amarp/dxr2`
- Current v0.30 implementation: `d24080f5` — template precision applies consistently to static CLAS and animated templates
- Temporary v0.29 sample branch: `user/amarp/dxr2-v029-sample`
- Temporary v0.29 sample implementation: `c2b26c19`
- Both branches contain fetched `origin/master` at `357ade6e`; merge commands returned `Already up to date`
- Never push this repository to Microsoft `origin`; push only to Amar's configured fork when authorized

Start by fetching the internal runtime branch and reading `DXR2_AGENT_HANDOFF.md`; it contains the exact RTX 5070 resume checklist and validation artifacts.
