# Δ₁ Tesseract Field Engine

A real-time visual field engine based on Δ₁ folding geometry, Fibonacci phase offsets, and radial tesseract projection.  
The engine transforms microphone input into a multi-layer harmonic field displayed as a 720-spoke radial bloom with memory-depth accumulation and bowl-mode inversion.

## Features
- Δ-fold audio folding using λ₁ and λ₂ phase offsets
- 720 radial spokes with per-layer jitter to eliminate symmetry seams
- 64-layer memory buffer for persistent fold-field trails
- Bowl inversion mode for inverted-radii harmonic geometry
- Fully standalone HTML + JS (no dependencies)
- GitHub Pages ready

## Demo
After pushing this repo, enable GitHub Pages:
**Settings → Pages → Deploy from branch → main → /root**  
Your Δ₁ field will appear at:  
`https://<your-username>.github.io/delta1-tesseract-field/`

## Files
- `index.html` — Full engine, microphone input enabled
- `fallback.html` — Synthetic harmonic mode for sandboxes (Wix etc.)

## Notes
Mic input requires HTTPS and cannot run inside sandboxed iframes (such as Wix's HTML Embed), so host directly via GitHub Pages for full functionality.
