# CLAIM BOUNDARY

## What SafeDyn Claims

SafeDyn guarantees:

1. **No bypass**: Every action written to the actuator was evaluated by
   `CertifiedAccept`.
2. **Fail-closed**: Uncertified actions are never written; certified fallback
   (STOP) is used instead.
3. **B_cert ≥ B_plan**: The certified tube is always more conservative than
   the planning tube.

## What SafeDyn Does NOT Claim

1. **No collisions**: The certificate chain uses geometric models (tubes,
   axis-aligned bounding boxes, velocity-uncertainty models). Real-world
   tracking errors, sensor noise, and model mismatch may cause collisions.
2. **Formal safety guarantee**: This is a runtime guard, not a formally verified
   system. The assertions are enforced by tests, not by formal methods.
3. **Generalization to all environments**: Parameters (B_cert margin, fallback
   priorities) are tuned for the VLN-CE / MP3D evaluation setup.
4. **Navigation performance**: SafeDyn only provides safety certification.
   Navigation success depends on the underlying VLN policy.

## Paper-Specific Claims

The paper describes additional experiments (RQs, tables, figures) that are
not part of this method-only release. Those experiments required restricted
data, external weights, and large-scale Habitat execution. They are not
reproducible from this repository alone.

## Claim Boundary Summary

| Claim | Support | Reproducible? |
|-------|---------|---------------|
| CertifiedAccept evaluates actions | Code + tests | Yes |
| Fail-closed: no uncertified action written | Code + tests | Yes |
| B_cert ≥ B_plan invariant | Code + tests | Yes |
| Collision-free in all scenarios | No formal proof | No |
| Generalization beyond VLN-CE scenarios | Not claimed | No |
| Paper RQ results reproduced | Requires external data | No |
