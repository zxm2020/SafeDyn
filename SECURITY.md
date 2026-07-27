# SECURITY

## Reporting

Report vulnerabilities privately to the paper authors (see contact information).
Do not open public GitHub issues.

## Scope

SafeDyn is a research prototype. It is not hardened for production deployment.

## Known Limitations

- Geometric safety models (tubes, AABBs) are approximations.
- Kalman filter tracking assumes linear dynamics and Gaussian noise.
- CBF-QP lite is a linearized approximation.
- Fail-closed provides software-level safety but no hardware watchdog redundancy.
