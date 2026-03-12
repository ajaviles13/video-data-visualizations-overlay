# 🎬 Video Data Visualizations Overlay
## Transparent Heart Rate & Temperature Overlay for Wellness Videos

A professional Python system that creates **transparent overlay videos** with biometric data. Perfect for **cold plunge**, **sauna**, and **cold shower** sessions recorded with [PlungePalz](https://plungepalz.com) or similar health tracking apps. Output is a WebM with alpha channel that you can composite over any video in your editor.

---

## ✨ Features

- **Transparent Output**: 1080×1920 WebM with alpha channel—composite over any video
- **No Input Video Required**: Renders overlay onto blank transparent canvas
- **PlungePalz Integration**: Designed for cold plunge, sauna, and cold shower data
  - Cold Plunge: Heart rate & temperature recorded every 1 second
  - Sauna & Cold Shower: Heart rate & temperature recorded every 5 seconds
- **Exact Design Match**: Pixel-perfect positioning (1080×1920 portrait)
- **Pulsing Heart Animation**: Heart icon pulses in sync with actual BPM (±15% size variation)
- **Professional Typography**: Poppins-Bold font with black stroke outline
- **Smooth Rendering**: Progress tracking with configurable FPS
- **Easy to Use**: Simple command-line interface

---

## 📐 Design Specifications

**Canvas Dimensions**: 1080 × 1920 pixels (portrait 9:16)

### Heart Icon ❤️
- **Position**: 21.1% from left, 75.5% from top
- **Size**: 15.7% of video width (198px at reference)
- **Color**: Red (#FF6B6B)
- **Animation**: Pulsing ±15% synced to BPM

### Text Display
- **Position**: 38.3% from left, 76.4% from top
- **Font**: Poppins-Bold
- **Size**: 9.5% of video width (120px at reference)
- **Color**: White (#FFFFFF)
- **Stroke**: Black (#000000), 8-10px thickness
- **Format**: "{heart_rate} BPM" (e.g., "88 BPM")

---

## 🚀 Quick Start

### 1. Installation

```bash
cd VideoTextOverlay_HeartRateData

# Install dependencies
pip install -r requirements.txt

# Download required assets (font + heart icon)
python src/download_assets.py
```

### 2. Prepare Your Data

**No input video needed**—the overlay renders onto a transparent 1080×1920 canvas.

**Prepare heart rate CSV**:

Create or update `input/heartrate.csv` with your data:

```csv
heart_rate
72
75
78
82
85
...
```

Or with timestamps:

```csv
timestamp,heart_rate
0,72
1,75
2,78
3,82
4,85
...
```

### 3. Validate Your Data (Optional)

```bash
python src/validate_csv.py
```

Output example:
```
✓ CSV Format: Valid
✓ Columns: heart_rate
✓ Total Rows: 210 (3:30 duration)
✓ Heart Rate Range: 72-115 BPM
✓ Average: 88 BPM
```

### 4. Generate Transparent Overlay

```bash
# Heart rate only
python src/overlay_video.py

# Heart rate + temperature (dual charts)
python src/overlay_second_intervals_video_hr_and_temp_data.py
```

Output: `output/transparent_overlay.webm` — a WebM with alpha you can layer over any video in your editor.

---

## 🏔️ Using with PlungePalz

This tool was designed with [PlungePalz](https://plungepalz.com) in mind - the ultimate cold plunge, sauna, and cold therapy tracking app.

### Data Recording Intervals

PlungePalz records biometric data at different intervals:
- **Cold Plunge Sessions**: ✅ 1-second intervals (heart rate & temperature)
- **Sauna Sessions**: 5-second intervals (heart rate & temperature)
- **Cold Shower Sessions**: 5-second intervals (heart rate & temperature)

### Export Your PlungePalz Data

1. Open your PlungePalz session
2. Export the session data as CSV
3. The CSV will include timestamps, heart rate, and temperature
4. Use the CSV directly with this tool!

### Tips for Best Results

- **Cold Plunge Videos**: Perfect 1:1 match - no interpolation needed!
- **Sauna/Cold Shower Videos**: Data is recorded every 5 seconds, so the overlay will update every 5 seconds (or you can interpolate for smoother animation)
- **Video Recording**: Start recording your video at the same time you start your PlungePalz session
- **Synchronization**: Trim your video to match the CSV data duration for perfect sync

---

## 📖 Detailed Usage

### Download Assets

```bash
# Download font and create heart icon
python src/download_assets.py
```

This script:
- Downloads Poppins-Bold.ttf from Google Fonts
- Creates a beautiful red heart icon (198x198px PNG)
- Validates all downloads
- Shows clear success/error messages

### Validate CSV

```bash
# Basic validation
python src/validate_csv.py

# Validate specific file
python src/validate_csv.py --csv input/mydata.csv

# Check against expected duration
python src/validate_csv.py --duration 210
```

Validates:
- ✓ Correct columns (heart_rate required)
- ✓ No missing values
- ✓ Realistic BPM range (40-220)
- ✓ Shows statistics and distribution
- ✓ Preview of first/last 5 rows

### Generate Overlay

```bash
# Heart rate only (defaults)
python src/overlay_video.py

# Heart rate + temperature
python src/overlay_second_intervals_video_hr_and_temp_data.py

# Custom output
python src/overlay_video.py --output output/my_overlay.webm --csv input/mydata.csv
```

**overlay_video.py options**:
- `--output`, `-o`: Output WebM file (default: `output/transparent_overlay.webm`)
- `--csv`, `-c`: Heart rate CSV file (default: `input/heartrate.csv`)
- `--fps`: Frames per second (default: 30)
- `--width`, `--height`: Canvas size (default: 1080×1920)

**overlay_second_intervals_video_hr_and_temp_data.py** (adds `--hr-csv`, `--temp-csv`)

---

## 📂 Project Structure

```
VideoTextOverlay_HeartRateData/
├── input/
│   ├── heartrate.csv          # Your heart rate data (add this)
│   └── temperature_data.csv   # Temperature data (for dual overlay)
├── output/
│   └── transparent_overlay.webm  # Transparent overlay (generated)
├── assets/
│   ├── fonts/
│   │   └── Poppins-Bold.ttf   # Downloaded by download_assets.py
│   └── images/
│       └── heart.png          # Created by download_assets.py
├── src/
│   ├── download_assets.py     # Download fonts & create icons
│   ├── overlay_video.py       # Heart rate overlay (transparent)
│   ├── overlay_second_intervals_video_hr_and_temp_data.py  # HR + temp overlay
│   └── validate_csv.py       # CSV validation tool
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

---

## 📊 CSV Format

### PlungePalz Data Export

When exporting data from PlungePalz:
- **Cold Plunge**: Data recorded every 1 second (heart rate & temperature)
- **Sauna**: Data recorded every 5 seconds (heart rate & temperature)
- **Cold Shower**: Data recorded every 5 seconds (heart rate & temperature)

Export your session data as CSV and use it directly with this tool!

### Option 1: Simple Format (Recommended)

One heart rate value per second:

```csv
heart_rate
72
75
78
82
...
```

### Option 2: With Timestamps

Explicit timestamps (in seconds):

```csv
timestamp,heart_rate
0,72
1,75
2,78
3,82
...
```

### Option 3: With Temperature (Coming Soon)

For PlungePalz sessions with temperature data:

```csv
timestamp,heart_rate,temperature
0,72,55.2
1,75,54.8
2,78,54.5
3,82,54.1
...
```

**Requirements**:
- ✓ Must have `heart_rate` column
- ✓ Values between 40-220 BPM recommended
- ✓ No missing values
- ✓ One row per second of video (or interpolate 5-second data)

---

## 🎨 How It Works

### Design Scaling

The system uses **percentage-based positioning** to scale perfectly to any video resolution:

```python
# Reference dimensions: 1261x2242 (portrait 9:16)
heart_left = 21.1% of video width
heart_top = 75.5% of video height
heart_size = 15.7% of video width

text_left = 38.3% of video width
text_top = 76.4% of video height
font_size = 9.5% of video width
```

### Pulse Animation

Heart pulses using sine wave synchronized to actual BPM:

```python
pulse_scale = 1.0 + 0.15 * sin(2π * time * BPM/60)
# Results in 0.85x to 1.15x scale variation
```

This creates a smooth, realistic heartbeat effect!

### Text Rendering

Professional text with stroke outline:

1. **Stroke**: Render text 8 times in circular pattern (black)
2. **Main Text**: Render white text on top
3. **Blend**: Alpha composite onto video frame

---

## 🔧 Troubleshooting

### "FFmpeg not found"

**Problem**: FFmpeg is required to encode WebM with alpha

**Solutions**:
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg` or `yum install ffmpeg`

### "CSV must have 'heart_rate' column"

**Problem**: CSV missing required column

**Solution**:
```csv
heart_rate   ← Must have this column name
72
75
78
```

### "Font not found"

**Problem**: Poppins-Bold.ttf not downloaded

**Solution**:
```bash
python src/download_assets.py
```

### "Values outside typical range"

**Problem**: Heart rate values < 40 or > 220 BPM

**Solutions**:
- Check for typos in CSV
- Verify data is in BPM (not percentage)
- Confirm sensor readings are accurate

### Slow Processing

**Tips**:
- Processing time: ~0.5-1 second per video second
- Use SSD for faster I/O
- Close other applications
- Lower resolution videos process faster

---

## 🎯 Examples

### Cold Plunge Session (PlungePalz)

```bash
# Export your PlungePalz cold plunge data as CSV (1-second intervals)
python src/overlay_second_intervals_video_hr_and_temp_data.py \
  --hr-csv input/plunge_session.csv \
  --temp-csv input/plunge_temp.csv \
  --output output/cold_plunge_overlay.webm
# Then composite the .webm over your cold plunge video in your editor
```

### Heart Rate Only

```bash
python src/overlay_video.py --csv input/heartrate.csv --output output/hr_overlay.webm
```

### Custom Canvas Size

```bash
# Default is 1080×1920; override if needed
python src/overlay_video.py --width 1080 --height 1920 --output output/overlay.webm
```

---

## 📋 Requirements

- Python 3.8+
- **FFmpeg** (for WebM encoding with alpha)
- OpenCV 4.8+
- NumPy 1.24+
- Pandas 2.0+
- Pillow 10.0+
- tqdm 4.65+
- requests 2.31+
- matplotlib, scipy (for chart rendering)

All Python dependencies:
```bash
pip install -r requirements.txt
```
FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

---

## 🎥 Video Formats

**Input**: None required—overlay renders onto transparent canvas.

**Output Format**:
- WebM (VP9 codec with alpha channel)
- Resolution: 1080×1920 (portrait)
- Transparent background—composite over any video in your editor
- **Requires FFmpeg** (e.g., `brew install ffmpeg` on macOS)

---

## 💡 Tips & Best Practices

### For Best Results

1. **Video Quality**: Use high-resolution video (1080p or higher)
2. **CSV Accuracy**: Ensure heart rate data matches video duration
3. **Validation**: Always run `validate_csv.py` first
4. **Backup**: Keep original video files
5. **Testing**: Test with short video clips first

### CSV Preparation

1. Export data from your fitness tracker
2. Convert to CSV with 1 row per second
3. Validate with `validate_csv.py`
4. Process video

### Primary Use Cases (PlungePalz)

- 🧊 **Cold Plunge Sessions**: Track heart rate and temperature recovery (1-second intervals)
- 🔥 **Sauna Sessions**: Monitor heat exposure and cardiovascular response (5-second intervals)
- 🚿 **Cold Shower Sessions**: Display cold therapy adaptation (5-second intervals)

### Additional Use Cases

- 🏃 **Workout Videos**: Show heart rate during exercise
- 🧘 **Meditation**: Display resting heart rate
- 🚴 **Cycling**: Overlay training data
- 🏊 **Swimming**: Show pool workout intensity

---

## 🐛 Known Issues

1. **Large Videos**: Files > 1GB may take 10-30 minutes to process
2. **Audio**: Currently copies video only (no audio processing)
3. **4K Video**: May be slow on older machines

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

Feel free to use, modify, and distribute this tool as needed!

---

## 🤝 Support

For issues or questions:

1. Check this README
2. Run `validate_csv.py` to verify data
3. Check error messages for specific issues
4. Ensure all assets downloaded (`download_assets.py`)

---

## 🎉 Credits

- **PlungePalz**: Designed for integration with [PlungePalz](https://plungepalz.com) cold plunge, sauna, and cold therapy tracking
- **Font**: Poppins-Bold by Indian Type Foundry (Google Fonts)
- **Heart Icon**: Custom created using PIL
- **Design Reference**: Based on Figma/React Native specifications
- **Author**: AJ Aviles ([@ajaviles13](https://github.com/ajaviles13))

---

## 📸 Example Output

```
📹 Transparent Overlay Canvas:
   Resolution: 1080x1920 (portrait)
   FPS: 30.00
   Total Frames: 6300
   Duration: 210.00s (3.50 min)
   Output Format: WebM with alpha (VP9)

💓 Heart Rate Data:
   Data points: 210
   Duration: 210s (3.50 min)
   Range: 72-115 BPM

🎯 Overlay Positions:
   Heart: (228, 1450) size=170px
   Text: (413, 1469) font=103px
   Stroke: 9px

🎬 Processing video...
   Output: output/transparent_overlay.webm

Rendering: 100%|██████████| 6300/6300 [05:15<00:00, 21.54frames/s]

🎬 Encoding transparent WebM video...

✅ Processing complete!
   Processed 6300 frames
   Output saved to: output/transparent_overlay.webm

🎉 SUCCESS!
```

---

**Enjoy your professional heart rate overlay videos!** 🎬❤️
