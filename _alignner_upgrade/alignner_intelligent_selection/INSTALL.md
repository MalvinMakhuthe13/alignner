# Installation

1. Download and extract this ZIP into a temporary folder.
2. In your local clone of `alignner`, make a backup:
   ```bash
   copy app.py app.py.backup
   ```
   (PowerShell: `Copy-Item app.py app.py.backup`)
3. Copy the included `app.py` over the repository's `app.py`.
4. Copy `src/image_selector.py` into the repository's `src/` folder.
5. Keep the existing `src/aligner.py` and `requirements.txt` unchanged.
6. Install dependencies if needed:
   ```bash
   pip install -r requirements.txt
   ```
7. Run:
   ```bash
   streamlit run app.py
   ```

Selection is OFF by default. Enable **Intelligent selection → Enable conservative duplicate removal** to test it.

## Git commands

```bash
git status
git add app.py src/image_selector.py README_UPGRADE.md
git commit -m "Add conservative intelligent image selection"
git push origin main
```

If you are working on a feature branch, replace `main` with that branch name.

## Rollback
Restore `app.py.backup` and delete `src/image_selector.py`.
