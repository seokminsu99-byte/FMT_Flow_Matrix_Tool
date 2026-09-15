# Trusted local research inputs only

FMT is a desktop research tool, not a sandbox or a hardened upload service.
Open only files from sources you trust. In particular, the inherited model
loader uses `torch.load(..., weights_only=False)` for NumPy-containing legacy
checkpoints, and gold-label loading allows pickled object metadata. Such files
can execute code when loaded. Do not download arbitrary models or label files
and place them in FMT's data/resource directories. No pretrained files are shipped.

The public cleanup does not silently import older NFMAT profiles. A future safe
serialization migration needs explicit backward-compatibility tests and is not
claimed complete in this release. Background processing and GUI error handling
do not make hostile input safe.

The PLENA path points to a user-trusted local executable. Do not execute an
unverified binary. Do not commit credentials, municipal GIS, personal paths,
settings, models, private datasets or generated outputs. Report security issues
to the repository maintainer without posting sensitive payloads in public issues.
