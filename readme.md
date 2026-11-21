# Camera Stabilizer for OBS

A high-performance video stabilization plugin for OBS Studio designed to eliminate vibration and jitter from fixed PTZ cameras in high-vibration environments (such as churches with metal structures). Implements optical flow motion estimation combined with a high-pass baseline filter and adaptive cropping to maintain video quality while removing unwanted shake.

## What's new in v2

Inspired by LiveVisionKit, rebuilt for the current OBS (via Virtual Camera):

- **Rotation correction** - similarity model (translation + rotation), fitted with RANSAC. On the realistic benchmark rotation alone was responsible for most of the residual shake v1 left behind.
- **Robust tracking** - persistent feature tracks spread over an 8x6 grid, forward-backward LK check, CLAHE for dark church interiors. People walking in front of the camera are rejected as outliers.
- **Look-ahead smoothing** - an optional delay of N frames (default 4, ~133ms) enables a centered, zero-lag Gaussian filter before the high-pass baseline. Set the same delay as *Sync Offset* on the audio in OBS (N x 33ms at 30fps).
- **Single warp** - correction and crop zoom in one `warpAffine` (sharper, faster), and a fixed crop by default so the picture never "breathes".

Benchmark (`python bench/benchmark.py`, 720p, 8/13Hz shake + 0.35deg roll + sensor noise + moving person):

| | frame-to-frame jitter Y | rotation jitter | ms/frame |
|---|---|---|---|
| raw input | 12.48 px | 0.262 deg | - |
| v1 | 2.39 px | 0.262 deg | 7.2 |
| v2 (look-ahead 4) | **0.19 px** | **0.024 deg** | 4.9 |
| v2 (look-ahead 0, no delay) | 0.67 px | 0.067 deg | 5.4 |

## Profiles & recommended settings

The GUI has a **Perfil** bar: pick a preset or one of your profiles, then **Salvar** / **Salvar como...** / **Excluir**. Profiles store every field (source, resolution, sliders, checkboxes) in `%APPDATA%\CameraStabilizer\profiles.json`, and the last used profile is reopened on startup. Built-in presets are read-only.

On first run the app also creates an editable profile called **Igreja** (recommended settings, already selected), shipped in `config/default_profiles.json` and bundled by `build.bat`. It is created only once: if you edit or delete it, your choice is kept. To ship different starting profiles, edit that file before building.

| Preset | Smoothing | Crop | Look-ahead | When to use |
|---|---|---|---|---|
| Recomendado - camera fixa | 1.5 s | 90% | 4 | Default for a fixed/PTZ camera on a vibrating structure |
| Tremor forte | 2.0 s | 85% | 8 | Shake still visible, or borders get hit |
| Sem atraso | 1.0 s | 90% | 0 | You don't want to touch OBS audio sync |
| PTZ com muito movimento | 0.6 s | 92% | 4 | Frequent pans/zooms during the service |

All presets: sensitivity 100, rotation on, intentional-motion detection on, adaptive crop off. Remember: look-ahead N frames => OBS audio *Sync Offset* = N x 33 ms at 30 fps.

## Key Features

- Optical flow-based motion estimation with frame downsampling for low latency
- High-pass baseline filter that cancels vibration while letting intentional motion through
- Adaptive crop management that grows when needed, eases back gradually
- Distinguish vibration from intentional pan/tilt/zoom movements
- Live preview with debug overlay (motion vectors, FPS, processing time, crop ratio)
- Sensitivity, smoothing and crop limits configurable live from the GUI, with no restart
- Low latency: average 10ms per 1080p frame (well below 33ms budget for 30fps)
- Runs as standalone app publishing to OBS Virtual Camera (isolated from OBS crashes)
- Automatic reconnect: if the capture card drops out or processing fails, it reopens itself

## System Requirements

### Hardware

- Windows 10 or later (64-bit)
- OBS Studio 32.1.2 or newer
- Capture card / camera outputting 1080p @ 30fps via USB or similar interface
- Processor capable of 10-15% single-core utilization

### Software (Pre-packaged in Executable)

- Python 3.10+ (embedded in .exe)
- OpenCV 4.8+
- NumPy 1.24+
- PyVirtualCam 0.11+
- Pillow 9.0+
- Tkinter (for GUI)

## Installation

### Windows (Recommended)

1. Download or clone the repository
2. Navigate to the project folder
3. Double-click `build.bat` to generate the executable
   - This will create a Python virtual environment
   - Install all dependencies
   - Package everything into `dist\CameraStabilizer.exe`
4. The executable is ready to run immediately (no Python installation required)

### From Source (Alternative)

Requires Python 3.10+ installed on your system:

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python app.py
```

## Quick Start

### GUI Mode (Recommended)

1. Run `CameraStabilizer.exe`
2. Configure settings:
   - **Source**: Camera index (usually 0) or video file path
   - **Resolution**: Width, height, and FPS (default 1920x1080 @ 30fps)
   - **Sensitivity**: 0-100 slider, the fraction of detected shake actually cancelled (100 = fully locked)
   - **Smoothing**: 0.2-3.0s, how slowly the "home position" tracks the camera. Motion slower than this is kept, faster is cancelled
   - **Max Crop**: 80-98%, the smallest fraction of the picture the stabilizer may keep when it needs margin
   - **Detect Intentional Motion**: Enable to preserve pan/tilt/zoom movements
   - **Show Debug Overlay**: Display real-time motion vectors and performance metrics
3. Click "Start"
4. Add the resulting virtual camera to OBS:
   - Add source -> Video Capture Device
   - Select "OBS Virtual Camera"

### Command-Line Mode

For automated or headless deployment:

```bash
python camera_to_obs.py 0 --width 1920 --height 1080 --fps 30
```

Parameters:

- `source`: Camera index (0, 1, ...) or video file path
- `--width`: Frame width (default 1920)
- `--height`: Frame height (default 1080)
- `--fps`: Frames per second (default 30)

Configuration is read from `config/default_settings.json`.

## Configuration

Edit `config/default_settings.json` to adjust default parameters:

```json
{
  "strength": 1.0,
  "smoothing_s": 1.0,
  "max_keep_ratio": 0.98,
  "min_keep_ratio": 0.85,
  "detect_intentional_motion": true
}
```

### Parameter Reference

| Parameter                   | Range        | Description                                                |
| --------------------------- | ------------ | ---------------------------------------------------------- |
| `strength`                  | 0.0-1.0      | Fraction of detected shake cancelled (1.0 = fully locked)  |
| `smoothing_s`               | 0.2-3.0      | Baseline time constant in seconds; motion slower than this is preserved |
| `max_keep_ratio`            | 0.80-1.00    | Least-cropped state, used when the picture is calm         |
| `min_keep_ratio`            | 0.80-0.95    | Floor: never crop beyond this limit                        |
| `detect_intentional_motion` | true/false   | Enable pan/tilt/zoom preservation                          |
| `stabilize_rotation`        | true/false   | Correct camera roll (recommended)                          |
| `stabilize_zoom`            | true/false   | Correct scale jitter (off: would fight PTZ zoom)           |
| `adaptive_crop`             | true/false   | Zoom in only when needed (off = fixed crop at `min_keep_ratio`) |
| `lookahead_frames`          | 0-15         | Output delay for zero-lag smoothing; match OBS audio offset |

### Sensitivity Slider (GUI Only)

Maps directly to `strength`, the fraction of the detected deviation that gets cancelled:

- 0 = stabilization off (passthrough)
- 50 = roughly half the shake removed
- 100 = fully locked (default)

Measured on a simulated 8Hz/13Hz vertical shake, the slider moves residual
shake through 0% / 23% / 47% / 72% / 97% suppression across its travel, so
every position is visibly different.

## Performance Specifications

Tested on 1080p30 video from various capture cards:

- Average processing time: 10.2ms per frame
- Peak processing time: 11.9ms per frame
- Vertical shake suppression: 97% on a simulated 8Hz/13Hz metal-structure shake
- Crop actually used for that shake: 93% of the frame retained
- Frame rate: Maintains 30fps without dropping frames
- Crop overhead: Minimal (downsampled optical flow tracking)
- Memory: <150MB total

## How It Works

### Motion Detection

1. Each frame is downsampled to 25% for fast optical flow computation
2. Lucas-Kanade optical flow tracks corner features between frames
3. Median displacement (not mean) provides robust global motion estimate
4. Motion is scaled back up to full resolution

### Vibration vs. Intentional Motion

Distinguishes between:

- **Vibration**: Back-and-forth tremor (5-20Hz, net displacement ~= 0 over short window)
- **Pan/Tilt/Zoom**: Sustained movement in one direction (< 2Hz, consistent direction)

Ratio of net displacement to total path length determines which filter strength applies:

- Pure oscillation (ratio ~0): baseline held slow, the shake is cancelled
- Sustained motion (ratio ~1): baseline snaps to 0.12s so the pan passes through

Measured effect: a 118px pan is preserved at 95% with detection on, 63% with it off.

### Adaptive Cropping

Crop margin dynamically adjusts to motion demands:

- Grows quickly (2% per frame) when motion needs more headroom
- Releases slowly (0.4% per frame) once motion calms down
- Bounded between floor (80%) and ceiling (97%)
- Prevents visual "pumping" when crop edges flicker

### Baseline Filtering (High-Pass)

The camera is bolted to a vibrating structure, not handheld: its true position
is a fixed "home" that the vibration oscillates around. The stabilizer tracks
that home position with a slow exponential baseline and cancels everything
above it, which is the high-pass filter the original spec called for.

- Baseline: exponential, 1.0s time constant by default
- Correction: `strength * (baseline - trajectory)`, clamped to the crop margin
- Anti-windup: when the correction hits the crop limit the baseline is pulled
  to match, so the picture never parks against the crop edge and stays there

An earlier version used a 4-state constant-velocity Kalman filter here. Its
velocity state is excited by the vibration, so the estimate chases the shake
instead of ignoring it; measured across its entire usable tuning range it
capped out between 72% and 92% suppression. The position-only baseline
measures 97% with a single intuitive knob.

## OBS Integration

The stabilizer runs as a separate process and publishes to the OBS Virtual Camera:

```
[Capture Card] -> Stabilizer Process -> [Virtual Camera] -> [OBS]
                  (Isolated)
```

Benefits:

- If stabilizer crashes, OBS remains live
- Can restart without stopping broadcast
- Reconnects on its own if the capture card drops out mid-service
- GUI runs independently for parameter tuning
- No modifications to OBS installation required

### Setup in OBS

1. Ensure OBS Virtual Camera is installed (included with OBS 32.1.2+)
2. Start `CameraStabilizer.exe` and click "Start"
3. In OBS, add source:
   - Scene -> + (Add Source)
   - Video Capture Device
   - Create New -> Name it (e.g., "Stabilized Camera")
   - Device: "OBS Virtual Camera"
   - Click OK
4. Adjust preview in OBS as needed

## Troubleshooting

### "nao consegui abrir: 0" (Cannot open camera)

- **Windows**: Check Settings -> Privacy -> Camera. The Python process needs permission.
- Verify camera index with OBS's own Video Capture Device to confirm it works
- Try `--width 1280 --height 720` if 1080p is not supported

### The picture froze mid-service

The stabilizer reopens the capture source on its own every 2 seconds after a
dropout, and the status line shows "reconectando..." while it retries. Current
slider values are re-applied to the new session, so tuning is not lost.

Recovery is not automatic in one case: a capture card that hangs without
returning an error leaves the read blocked, and nothing times it out. If that
happens, press Parar and Iniciar.

### No virtual camera appears in OBS

- Ensure OBS 32.1.2+ is installed (provides Virtual Camera driver)
- Restart OBS after starting the stabilizer
- Check that no other app is currently using the Virtual Camera
- If still missing, reinstall OBS (registers the camera driver on install)

### High CPU usage or lag

- Reduce resolution (try 1280x720 instead of 1920x1080)
- Lower the sensitivity slider (CPU cost is dominated by optical flow, not the filter)
- Disable debug overlay (reduces rendering overhead)

### Frames look blurry or cropped too much

- Lower "Max Crop" (e.g. 85%) so the stabilizer has more margin to work with
- Reduce sensitivity if the crop opens further than you want
- Verify capture card supports MJPEG codec (more efficient)

### Motion detection too aggressive / too weak

- Adjust the sensitivity slider, or edit `strength` in `config/default_settings.json`
- If pan/tilt is being suppressed: enable "Detect Intentional Motion"
- If shake remains at sensitivity 100, the shake exceeds the crop budget: lower "Max Crop"
- Raise "Smoothing" for steadier framing, lower it to release intentional moves sooner

## Building from Source

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
pyinstaller --onefile --windowed --name CameraStabilizer ^
    --add-data "config;config" ^
    --collect-all cv2 ^
    --collect-all pyvirtualcam ^
    app.py
```

Executable appears in `dist\CameraStabilizer.exe`.

### macOS / Linux

Not officially supported (requires Tkinter and pyvirtualcam support for native cameras). Can be adapted for DirectShow alternatives, but OBS Virtual Camera is Windows-specific.

## Testing

Run self-checks to validate the build:

```bash
python test_stabilizer.py
```

Tests:

- Jitter reduction (synthetic shake)
- Vertical shake suppression (8Hz/13Hz metal-structure simulation)
- Live retuning takes effect without a restart
- Reconnect loop retries a dead camera and stops cleanly when asked
- Pan/tilt preservation (intentional motion)
- Latency budget compliance (< 33ms per frame)

## Project Structure

```
CameraStabilizer/
├── src/
│   └── stabilizer.py          # Core VideoStabilizer class
├── config/
│   └── default_settings.json  # Configuration template
├── app.py                      # GUI (Tkinter)
├── camera_to_obs.py           # Pipeline & CLI interface
├── demo.py                    # Simple side-by-side preview
├── test_stabilizer.py         # Self-checks
├── build.bat                  # Windows build script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Development Notes

### Optimization Strategy

The stabilizer prioritizes latency over CPU load:

1. Optical flow computed on 25% downsampled frames
2. Baseline filter is two exponential updates per frame (negligible)
3. Crop adjustments are frame-local (no multi-frame buffering)
4. No thread synchronization overhead (single-threaded pipeline)

Target: < 16ms latency for OBS compositing, leaving 17ms for system scheduling at 30fps.

### Future Enhancements

- C++ core for > 2x speedup (phase 4 of spec)
- Gyroscope fusion for hybrid stabilization (if PTZ camera supports it)
- FFT-based frequency analysis for automatic parameter tuning
- Per-region selective stabilization (keep subject sharp, stabilize edges)

## License

Proprietary - developed for church broadcast applications.

## Support

For issues, questions, or feature requests, refer to the spec document or contact the development team.

---

Version: 1.0
Last Updated: 2026-07-22
