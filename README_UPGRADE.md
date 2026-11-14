# Alignner Intelligent Selection Upgrade

This is a Git-ready upgrade for `MalvinMakhuthe13/alignner`.

## Included
- `src/image_selector.py`: conservative quality assessment and near-duplicate selector.
- `app.py`: integrated Streamlit UI with opt-in selection, report table and CSV manifest.
- `tests/test_image_selector.py`: basic smoke tests.
- `INSTALL.md`: installation and push instructions.

## Safety behaviour
- Selection is OFF by default.
- Source images are never modified or deleted.
- Blurry / low-feature images are flagged for review and kept.
- Only very strong near-duplicates are automatically excluded.
- The original order is preserved.
- The selector is a preflight optimiser, not a reconstruction engine and not a replacement for COLMAP.

## Important limitation
This first version uses explainable image heuristics (Laplacian sharpness, mean exposure, ORB feature count, perceptual hash and ORB descriptor matching). It does not yet guarantee reconstruction completeness. Benchmark selected sets against the full set before using the reduced set for client deliverables.
