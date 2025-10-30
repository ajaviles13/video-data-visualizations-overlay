# 🎬 Video Data Visualizations Overlay
## Real-Time Heart Rate & Temperature Display for Wellness Videos

A professional Python system for overlaying real-time biometric data on videos. Perfect for **cold plunge**, **sauna**, and **cold shower** sessions recorded with [PlungePalz](https://plungepalz.com) or similar health tracking apps. Features pixel-perfect design specifications with pulsing heart animation synchronized to actual BPM.

---

## ✨ Features

- **PlungePalz Integration**: Designed for cold plunge, sauna, and cold shower data
  - Cold Plunge: Heart rate & temperature recorded every 1 second
  - Sauna & Cold Shower: Heart rate & temperature recorded every 5 seconds
- **Exact Design Match**: Pixel-perfect positioning based on reference dimensions (1261x2242)
- **Pulsing Heart Animation**: Heart icon pulses in sync with actual BPM (±15% size variation)
- **Professional Typography**: Poppins-Bold font with black stroke outline
- **Smooth Rendering**: 60fps+ processing with progress tracking
- **Flexible Input**: Supports any video resolution (auto-scales overlay)
- **Easy to Use**: Simple command-line interface

---

## 📐 Design Specifications

**Reference Dimensions**: 1261 x 2242 pixels (portrait 9:16)

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

### 2. Prepare Your Files

**Add your video**:
```bash
# Place your video file in the input directory
cp /path/to/your/video.mp4 input/video.mp4
```

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

### 4. Process Video

```bash
python src/overlay_video.py
```

Your video will be saved to `output/video_with_hr.mp4`

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

### Process Video

```bash
# Use defaults
python src/overlay_video.py

# Custom input/output
python src/overlay_video.py \
  --input input/myvideo.mp4 \
  --output output/result.mp4 \
  --csv input/mydata.csv
```

**Options**:
- `--input`, `-i`: Input video file (default: `input/video.mp4`)
- `--output`, `-o`: Output video file (default: `output/video_with_hr.mp4`)
- `--csv`, `-c`: Heart rate CSV file (default: `input/heartrate.csv`)

---

## 📂 Project Structure

```
VideoTextOverlay_HeartRateData/
├── input/
│   ├── video.mp4              # Your video file (add this)
│   └── heartrate.csv          # Your heart rate data (add this)
├── output/
│   └── video_with_hr.mp4      # Processed video (generated)
├── assets/
│   ├── fonts/
│   │   └── Poppins-Bold.ttf   # Downloaded by download_assets.py
│   └── images/
│       └── heart.png          # Created by download_assets.py
├── src/
│   ├── download_assets.py     # Download fonts & create icons
│   ├── overlay_video.py       # Main processing script
│   └── validate_csv.py        # CSV validation tool
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

### "Could not open video file"

**Problem**: Video file not found or corrupted

**Solutions**:
- Ensure video is at `input/video.mp4`
- Try different video format (MP4, MOV, AVI)
- Check video is not corrupted

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
# Overlay on your cold plunge video
python src/overlay_video.py \
  --input input/cold_plunge.mp4 \
  --csv input/plunge_session.csv \
  --output output/cold_plunge_overlay.mp4
```

### Sauna Session (PlungePalz)

```bash
# Export your PlungePalz sauna data as CSV (5-second intervals)
# Note: May need to interpolate to 1-second intervals for smooth animation
python src/overlay_video.py \
  --input input/sauna_session.mp4 \
  --csv input/sauna_data.csv \
  --output output/sauna_overlay.mp4
```

### Cold Shower Session (PlungePalz)

```bash
# Export your PlungePalz cold shower data as CSV (5-second intervals)
python src/overlay_video.py \
  --input input/cold_shower.mp4 \
  --csv input/shower_data.csv \
  --output output/cold_shower_overlay.mp4
```

### Portrait Video (9:16 - TikTok/Instagram/Reels)

```bash
# Perfect for social media - matches reference design
python src/overlay_video.py \
  --input input/portrait.mp4 \
  --output output/portrait_hr.mp4
```

### Landscape Video (16:9 - YouTube)

```bash
# Auto-scales to landscape format
python src/overlay_video.py \
  --input input/landscape.mp4 \
  --output output/landscape_hr.mp4
```

---

## 📋 Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- Pandas 2.0+
- Pillow 10.0+
- tqdm 4.65+
- requests 2.31+

All dependencies installed via:
```bash
pip install -r requirements.txt
```

---

## 🎥 Video Formats

**Supported Input Formats**:
- MP4 (recommended)
- MOV
- AVI
- Any format supported by OpenCV

**Output Format**:
- MP4 (H.264 codec)
- Same resolution as input
- Same frame rate as input

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
📹 Video Information:
   Resolution: 1080x1920
   FPS: 30.00
   Total Frames: 6300
   Duration: 210.00s (3.50 min)

💓 Heart Rate Data:
   Data points: 210
   Duration: 210s (3.50 min)
   Range: 72-115 BPM

🎯 Overlay Positions:
   Heart: (228, 1450) size=170px
   Text: (413, 1469) font=103px
   Stroke: 9px

🎬 Processing video...
   Output: output/video_with_hr.mp4

Rendering: 100%|██████████| 6300/6300 [05:15<00:00, 21.54frames/s]

✅ Processing complete!
   Processed 6300 frames
   Output saved to: output/video_with_hr.mp4

🎉 SUCCESS!
```

---

**Enjoy your professional heart rate overlay videos!** 🎬❤️
