# Δ₁ Fold-Field Engine

Real-time microphone-driven visual engine featuring radial tesseract projection, Fibonacci-based folding, and bowl-mode inversion. A mesmerizing audio visualization built with pure HTML/JS.

## Features

- 🎤 **Real-time Microphone Input** - Live audio visualization using Web Audio API
- 🔢 **64-Layer Memory Depth** - Temporal audio buffer for rich, layered effects
- 🌟 **720 Radial Spokes** - High-resolution radial rendering system
- 📐 **Fibonacci Folding** - Mathematical folding based on golden ratio
- 🎭 **Bowl Inversion Mode** - Switch between concave and convex projections
- 🎨 **Dynamic Color Cycling** - HSL-based color system with smooth transitions
- 🔄 **Tesseract Projection** - 4D to 2D rendering for depth effects
- 🎵 **Synthetic Fallback** - Automatic fallback mode when microphone unavailable

## Usage

### GitHub Pages Deployment

Simply enable GitHub Pages in your repository settings and point it to the root directory. The engine will be accessible at:
```
https://[username].github.io/D1-Fold-Field-Engine/
```

### Local Testing

Open `index.html` directly in a modern web browser, or serve it locally:
```bash
python3 -m http.server 8000
# Then navigate to http://localhost:8000
```

### Controls

- **START MICROPHONE** - Activate real-time audio input (requires microphone permission)
- **SYNTHETIC MODE** - Fallback mode with generated waveforms
- **BOWL INVERSION** - Toggle between normal and inverted projection
- **Fold Intensity** - Adjust Fibonacci folding strength (0.1 - 5.0)
- **Rotation Speed** - Control angular velocity (0 - 5.0)
- **Color Cycle** - Modify hue rotation speed (0 - 2.0)

## Technical Details

- **Self-contained** - Single HTML file, no external dependencies
- **Memory Layers** - 64 circular buffers storing temporal audio data
- **Radial Spokes** - 720 segments for smooth circular rendering
- **Golden Ratio (φ)** - 1.618... used in Fibonacci calculations
- **Browser Compatibility** - Modern browsers with Web Audio API support
- **Performance** - Optimized with memoization and requestAnimationFrame

## Browser Requirements

- Modern browser with Web Audio API support (Chrome, Firefox, Safari, Edge)
- For microphone mode: HTTPS or localhost (required for getUserMedia)
- Canvas 2D rendering support

## License

See LICENSE file for details.
