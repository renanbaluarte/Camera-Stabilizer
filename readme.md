# Camera Stabilizer for OBS

A high-performance video stabilization plugin for OBS Studio designed to eliminate vibration and jitter from fixed PTZ cameras in high-vibration environments (such as churches with metal structures). Implements optical flow motion estimation combined with Kalman filtering and adaptive cropping to maintain video quality while removing unwanted shake.

## Key Features

- Optical flow-based motion estimation with frame downsampling for low latency
- Kalman filter smoothing with dynamic adjustment for intentional motion detection
- Adaptive crop management that grows when needed, eases back gradually
- Distinguish vibration from intentional pan/tilt/zoom movements
- Live preview with debug overlay (motion vectors, FPS, processing time, crop ratio)
- Configurable sensitivity and crop limits via GUI
- Low latency: average 10ms per 1080p frame (well below 33ms budget for 30fps)
- Runs as standalone app publishing to OBS Virtual Camera (isolated from OBS crashes)

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
   - **Sensitivity**: 1-100 slider (higher = more aggressive stabilization)
   - **Max Crop**: 80-95% (maximum image retention without cropping beyond this)
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
  "crop_ratio": 0.97,
  "min_crop_ratio": 0.8,
  "kalman_q": 0.001,
  "kalman_r": 0.1,
  "detect_intentional_motion": true
}
```

### Parameter Reference

| Parameter                   | Range        | Description                                                |
| --------------------------- | ------------ | ---------------------------------------------------------- |
| `crop_ratio`                | 0.80-0.99    | Initial crop margin (0.97 = keep 97% of frame)             |
| `min_crop_ratio`            | 0.80-0.95    | Floor: never crop beyond this limit                        |
| `kalman_q`                  | 1e-4 to 1e-1 | Process noise (lower = smoother, laggy; higher = snappier) |
| `kalman_r`                  | 0.01 to 1.0  | Measurement noise (how much to trust optical flow)         |
| `detect_intentional_motion` | true/false   | Enable pan/tilt/zoom preservation                          |

### Sensitivity Slider (GUI Only)

Maps 1-100 to Kalman Q on a logarithmic scale:

- 1 = maximum smoothing (may lag)
- 50 = balanced (default: ~0.003)
- 100 = maximum responsiveness (may over-correct)

## Performance Specifications

Tested on 1080p30 video from various capture cards:

- Average processing time: 10.2ms per frame
- Peak processing time: 11.9ms per frame
- Jitter reduction: 70% on synthetic shake
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

- Pure oscillation (ratio ~0): Full Kalman smoothing
- Sustained motion (ratio ~1): Reduced smoothing, movement passes through

### Adaptive Cropping

Crop margin dynamically adjusts to motion demands:

- Grows quickly (2% per frame) when motion needs more headroom
- Releases slowly (0.4% per frame) once motion calms down
- Bounded between floor (80%) and ceiling (97%)
- Prevents visual "pumping" when crop edges flicker

### Kalman Filtering

4-state Kalman filter tracks position and velocity:

- State: [x, y, vx, vy]
- Process noise (Q): Tuned per sensitivity setting
- Measurement noise (R): Adapts based on motion classification
- Smooths trajectory while preserving intentional movements

## OBS Integration

The stabilizer runs as a separate process and publishes to the OBS Virtual Camera:

```
[Capture Card] -> Stabilizer Process -> [Virtual Camera] -> [OBS]
                  (Isolated)
```

Benefits:

- If stabilizer crashes, OBS remains live
- Can restart without stopping broadcast
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

### No virtual camera appears in OBS

- Ensure OBS 32.1.2+ is installed (provides Virtual Camera driver)
- Restart OBS after starting the stabilizer
- Check that no other app is currently using the Virtual Camera
- If still missing, reinstall OBS (registers the camera driver on install)

### High CPU usage or lag

- Reduce resolution (try 1280x720 instead of 1920x1080)
- Lower sensitivity slider (higher Kalman smoothing = less CPU load)
- Disable debug overlay (reduces rendering overhead)

### Frames look blurry or cropped too much

- Reduce sensitivity (lower Q = more aggressive smoothing)
- Increase "Max Crop" limit to allow more dynamic adjustment
- Verify capture card supports MJPEG codec (more efficient)

### Motion detection too aggressive / too weak

- Adjust Kalman Q via sensitivity slider or edit `config/default_settings.json`
- If pan/tilt is being suppressed: enable "Detect Intentional Motion"
- Check `kalman_r` if measurement is inconsistent

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
2. Kalman filter is 4-state (lightweight)
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
