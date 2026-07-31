# Parking Tracker Storage Cleanup Design

## Objective

Reduce the parking tracker's local repository and Docker storage without losing its trained models, supplied training backgrounds, PostgreSQL database, uploaded media, or ability to run and test through Docker.

## Current State

- The repository is roughly 80 MB. Most local repository usage is active model weights.
- The Docker web image occupies 8.71 GB.
- The image is ARM64 but contains a CUDA-enabled PyTorch build even though CUDA is unavailable inside the container.
- Unused NVIDIA packages consume about 2.94 GB, Triton consumes about 667 MB, and CUDA support consumes about 27 MB.
- OpenCV is the headless package, and its shared library does not link to `libGL` or GLib; the explicit GUI/runtime packages add about 200 MB unnecessarily.
- PostgreSQL uses about 67 MB, uploaded media uses less than 1 MB, and collected static files use no measurable space.
- There are no unrelated Docker images, containers, volumes, or build caches left after the earlier system cleanup.

## Protected Data

The cleanup must preserve:

- `apps/cv/weights/detector.pth`
- `apps/cv/weights/recognizer.pth`
- `data/backgrounds/`
- `parkingtracker_postgres_data`
- `parkingtracker_media`
- `parkingtracker_staticfiles`
- The tracked training plots under `artifacts/cv-training/`
- Source code, migrations, configuration, tests, and environment files

No command may use `docker compose down -v` or otherwise remove the named volumes.

## Repository Cleanup

Delete only confirmed reproducible or obsolete artifacts:

- `apps/cv/weights/detector.pth.bak`
- `apps/cv/weights/recognizer.pth.bak`
- Duplicate untracked plots in `apps/cv/weights/`
- `data/recognizer_train.log`
- `.coverage`
- `.ruff_cache/`
- Python bytecode, test caches, and temporary/log files created during verification

The active checkpoints and supplied backgrounds are not reproducible enough to treat as disposable and remain untouched.

## Docker Optimization

The Docker builder will install the officially published ARM64 CPU-only PyTorch 2.12.0 and torchvision 0.27.0 wheels from `https://download.pytorch.org/whl/cpu` before installing the remaining requirements. The existing lower-bound requirements remain satisfied, while pip will no longer select CUDA-enabled wheels from the default index.

The runtime stage will stop installing `libgl1` and `libglib2.0-0`. The project already uses `opencv-python-headless`, and inspection confirmed that its `cv2` shared library does not require either library.

Development and test dependencies remain in the default Compose image so existing `docker compose exec web pytest` and `django_extensions` workflows continue to work. The production override continues to request runtime-only dependencies.

## Replacement Flow

1. Build the optimized web image while retaining the current image and all volumes.
2. Verify imports and confirm `torch.version.cuda` is `None`.
3. Run the project's test suite in the new image.
4. Recreate the Compose services without removing named volumes.
5. Verify migrations, container health, database connectivity, and application response.
6. Compare image and Docker disk usage with the baseline.
7. Remove only the superseded dangling image, replaced stopped container, build cache, and container logs.

If the build or verification fails, retain the current image and containers and make no volume changes.

## Verification Criteria

- Compose configuration validates successfully.
- `torch`, `torchvision`, and `cv2` import successfully.
- PyTorch reports the expected CPU build and no CUDA runtime.
- The full test suite passes.
- The web and database containers become healthy/running.
- Existing database and media volumes retain their identities and sizes.
- The health endpoint responds successfully.
- The optimized image is materially smaller than 8.71 GB.
- Git contains only the intended Docker dependency/configuration changes; generated cleanup artifacts remain ignored.

## Expected Result

The CUDA and unneeded system-library removal should reclaim approximately 3.8 GB or more. Repository cleanup contributes about 29 MB. Exact savings will be measured after Docker has removed the superseded image and compacted its storage.
