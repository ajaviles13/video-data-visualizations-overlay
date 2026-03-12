#!/usr/bin/env python3
"""
Professional Video Overlay System - Heart Rate Display
Matches exact Figma/React Native design specifications
"""

import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse
import math
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path


def create_text_with_stroke(
    heart_rate: int,
    font_bold_path: Path,
    font_regular_path: Path,
    font_size: int,
    text_color: tuple = (255, 255, 255),
    stroke_color: tuple = (0, 0, 0),
    stroke_width: int = 8
) -> np.ndarray:
    """
    Create text image with black stroke using PIL.
    Number is bold, "BPM" is regular weight.
    
    Args:
        heart_rate: Heart rate value (e.g., 88)
        font_bold_path: Path to Poppins-Bold.ttf
        font_regular_path: Path to Poppins-Regular.ttf
        font_size: Font size in pixels (scaled to video resolution)
        text_color: RGB tuple for text (default: white)
        stroke_color: RGB tuple for stroke (default: black)
        stroke_width: Stroke thickness in pixels
        
    Returns:
        RGBA numpy array with text and stroke
    """
    try:
        font_bold = ImageFont.truetype(str(font_bold_path), font_size)
    except Exception as e:
        print(f"⚠️  Warning: Could not load font {font_bold_path}: {e}")
        print(f"   Using default font instead")
        font_bold = ImageFont.load_default()
    
    try:
        font_regular = ImageFont.truetype(str(font_regular_path), font_size)
    except Exception as e:
        print(f"⚠️  Warning: Could not load font {font_regular_path}: {e}")
        print(f"   Using default font instead")
        font_regular = ImageFont.load_default()
    
    # Split text into number (bold) and " BPM" (regular)
    number_text = str(heart_rate)
    bpm_text = " BPM"
    
    # Get bounding boxes for both parts
    dummy_img = Image.new('RGBA', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    number_bbox = dummy_draw.textbbox((0, 0), number_text, font=font_bold)
    number_width = number_bbox[2] - number_bbox[0]
    number_height = number_bbox[3] - number_bbox[1]
    
    bpm_bbox = dummy_draw.textbbox((0, 0), bpm_text, font=font_regular)
    bpm_width = bpm_bbox[2] - bpm_bbox[0]
    bpm_height = bpm_bbox[3] - bpm_bbox[1]
    
    # Total dimensions
    total_width = number_width + bpm_width
    total_height = max(number_height, bpm_height)
    
    # Create image with padding for stroke
    padding = stroke_width * 2
    img_width = total_width + padding * 2
    img_height = total_height + padding * 2
    
    img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Calculate positions
    number_x = padding - number_bbox[0]
    number_y = padding - number_bbox[1]
    
    bpm_x = number_x + number_width
    bpm_y = padding - bpm_bbox[1]
    
    # Offsets for stroke
    offsets = [
        (-stroke_width, 0), (stroke_width, 0),
        (0, -stroke_width), (0, stroke_width),
        (-stroke_width, -stroke_width), (stroke_width, -stroke_width),
        (-stroke_width, stroke_width), (stroke_width, stroke_width)
    ]
    
    # Draw strokes for both parts
    for dx, dy in offsets:
        draw.text((number_x + dx, number_y + dy), number_text, font=font_bold, fill=stroke_color)
        draw.text((bpm_x + dx, bpm_y + dy), bpm_text, font=font_regular, fill=stroke_color)
    
    # Draw main text on top (white)
    draw.text((number_x, number_y), number_text, font=font_bold, fill=text_color)
    draw.text((bpm_x, bpm_y), bpm_text, font=font_regular, fill=text_color)
    
    # Convert to numpy array
    return np.array(img)


def overlay_image_alpha(
    background: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    convert_rgb_to_bgr: bool = False
) -> np.ndarray:
    """
    Overlay RGBA image with transparency onto BGR/BGRA background.
    
    Supports both BGR (3-channel) and BGRA (4-channel) transparent canvases.
    
    Args:
        background: BGR or BGRA background image
        overlay: RGBA overlay image (text or icon)
        x: X position (left edge)
        y: Y position (top edge)
        convert_rgb_to_bgr: If True, convert RGB channels to BGR (for PIL images)
        
    Returns:
        Combined BGR/BGRA image
    """
    if overlay.shape[2] != 4:
        return background
    
    # Convert RGB to BGR if needed (PIL images are RGB, OpenCV is BGR)
    if convert_rgb_to_bgr:
        overlay = overlay.copy()
        overlay[:, :, :3] = overlay[:, :, [2, 1, 0]]  # Swap R and B channels
    
    h, w = overlay.shape[:2]
    
    # Clip to frame boundaries
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        if x < 0 or y < 0:
            overlay = overlay[max(0, -y):, max(0, -x):]
            x = max(0, x)
            y = max(0, y)
            h, w = overlay.shape[:2]
        
        if x + w > background.shape[1]:
            overlay = overlay[:, :background.shape[1] - x]
            w = overlay.shape[1]
        
        if y + h > background.shape[0]:
            overlay = overlay[:background.shape[0] - y, :]
            h = overlay.shape[0]
    
    alpha = overlay[:, :, 3] / 255.0
    roi = background[y:y+h, x:x+w]
    num_channels = background.shape[2]
    
    for c in range(num_channels):
        roi[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
    
    background[y:y+h, x:x+w] = roi
    
    return background


def encode_transparent_video(
    frames_dir: Path,
    output_path: Path,
    fps: float
) -> bool:
    """
    Encode PNG frame sequence to WebM with alpha (VP9) using ffmpeg.
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg to create transparent video.")
        print("   Install: brew install ffmpeg  (macOS) or apt install ffmpeg (Linux)")
        return False
    
    input_pattern = str(frames_dir / "frame_%05d.png")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',
        '-auto-alt-ref', '0',
        '-b:v', '2M',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ FFmpeg encoding failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ FFmpeg encoding error: {e}")
        return False


def calculate_pulse_scale(current_time: float, bpm: float) -> float:
    """
    Calculate heart pulse scale factor based on current time and BPM.
    
    Args:
        current_time: Current video time in seconds
        bpm: Current heart rate in beats per minute
        
    Returns:
        Scale factor between 0.85 and 1.15
    """
    # Calculate pulse frequency (beats per second)
    beats_per_second = bpm / 60.0
    
    # Calculate phase in pulse cycle (0 to 2π)
    pulse_phase = (current_time * beats_per_second * 2 * math.pi) % (2 * math.pi)
    
    # Sine wave oscillation: 0.85 to 1.15 (±15% amplitude)
    pulse_scale = 1.0 + 0.15 * math.sin(pulse_phase)
    
    return pulse_scale


def create_heart_rate_chart(
    heart_rate_df: pd.DataFrame,
    current_time: float,
    chart_width: int,
    chart_height: int,
    video_duration: float
) -> np.ndarray:
    """
    Create a progressive heart rate chart that fills from left to right.
    
    Args:
        heart_rate_df: DataFrame with heart rate data
        current_time: Current video time in seconds
        chart_width: Width of chart in pixels
        chart_height: Height of chart in pixels (25% of video height)
        video_duration: Total video duration in seconds
        
    Returns:
        RGBA numpy array with chart overlay
    """
    # Get data up to current time
    max_time_idx = int(current_time)
    if max_time_idx < 1:
        max_time_idx = 1
    
    # Get heart rate data up to current time
    data_slice = heart_rate_df.iloc[:max_time_idx + 1].copy()
    
    if len(data_slice) < 2:
        # Not enough data yet, return transparent image
        return np.zeros((chart_height, chart_width, 4), dtype=np.uint8)
    
    # Get time points (0 to current_time)
    if 'timestamp' in data_slice.columns:
        time_points = data_slice['timestamp'].values
    else:
        time_points = np.arange(len(data_slice))
    
    hr_values = data_slice['heart_rate'].values
    
    # Get min and max for auto-scaling
    hr_min = heart_rate_df['heart_rate'].min()
    hr_max = heart_rate_df['heart_rate'].max()
    
    # Add some padding to the range
    hr_range = hr_max - hr_min
    hr_min_display = hr_min - hr_range * 0.05
    hr_max_display = hr_max + hr_range * 0.05
    
    # Create figure with transparent background
    dpi = 100
    fig_width = chart_width / dpi
    fig_height = chart_height / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    
    # Set semi-transparent black background for plot area
    ax.set_facecolor('#000000')
    ax.patch.set_alpha(0.7)  # Semi-transparent black
    
    # Lazy import matplotlib/scipy (slow to load)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    # Create smooth curve using spline interpolation
    if len(time_points) > 3:
        # Use spline for smooth curve
        x_smooth = np.linspace(time_points[0], time_points[-1], len(time_points) * 10)
        spl = make_interp_spline(time_points, hr_values, k=min(3, len(time_points) - 1))
        y_smooth = spl(x_smooth)
    else:
        x_smooth = time_points
        y_smooth = hr_values
    
    # Scale x-axis to match video duration (0 to video_duration)
    x_scaled = (x_smooth / video_duration) * video_duration
    
    # Plot the line
    line_color = '#FF6E65'
    ax.plot(x_scaled, y_smooth, color=line_color, linewidth=2.5, zorder=3)
    
    # Fill area under the curve with gradient effect
    # Create gradient by filling multiple times with decreasing alpha
    fill_color = '#FF6E65'
    ax.fill_between(x_scaled, hr_min_display, y_smooth, 
                     color=fill_color, alpha=0.6, zorder=2)
    
    # Set axis limits
    ax.set_xlim(0, video_duration)
    ax.set_ylim(hr_min_display, hr_max_display)
    
    # Remove all spines except left (for Y-axis labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Set Y-axis ticks to show only min and max
    ax.set_yticks([hr_min, hr_max])
    ax.set_yticklabels([f'{int(hr_min)}', f'{int(hr_max)}'], 
                       color='white', fontsize=10, fontweight='normal')
    
    # Remove grid
    ax.grid(False)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Adjust layout to remove ALL padding (full width)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    
    # Convert plot to image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    chart_img = np.frombuffer(buf, dtype=np.uint8)
    chart_img = chart_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    plt.close(fig)
    
    return chart_img


def get_heart_rate_at_time(df: pd.DataFrame, time_seconds: float) -> int:
    """
    Get heart rate value for a specific time.
    
    Args:
        df: DataFrame with heart rate data
        time_seconds: Time in seconds
        
    Returns:
        Heart rate value (interpolated if needed)
    """
    if 'timestamp' in df.columns:
        # Find closest timestamp
        idx = df['timestamp'].searchsorted(time_seconds)
        if idx >= len(df):
            idx = len(df) - 1
        elif idx > 0 and abs(df.iloc[idx-1]['timestamp'] - time_seconds) < abs(df.iloc[idx]['timestamp'] - time_seconds):
            idx = idx - 1
        return int(df.iloc[idx]['heart_rate'])
    else:
        # Use row index as timestamp (1 row = 1 second)
        idx = int(time_seconds)
        if idx >= len(df):
            idx = len(df) - 1
        return int(df.iloc[idx]['heart_rate'])


def process_video(
    output_video: Path,
    heart_rate_df: pd.DataFrame,
    font_path: Path,
    heart_icon_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: float = 30.0,
    include_chart: bool = True
) -> bool:
    """
    Create transparent overlay video with heart rate data.
    
    Renders overlays onto a blank transparent canvas (1080x1920) and exports
    as WebM with alpha channel for compositing over other videos.
    
    Args:
        output_video: Path to output video file (WebM with alpha)
        heart_rate_df: DataFrame with heart rate data
        font_path: Path to Poppins-Bold.ttf
        heart_icon_path: Path to heart.png
        width: Canvas width in pixels (default: 1080)
        height: Canvas height in pixels (default: 1920)
        fps: Frames per second (default: 30)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        duration = len(heart_rate_df)
        total_frames = int(duration * fps)
        
        print(f"\n📹 Transparent Overlay Canvas:")
        print(f"   Resolution: {width}x{height} (portrait)")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {duration:.2f}s ({duration/60:.2f} min)")
        print(f"   Output Format: WebM with alpha (VP9)")
        
        print(f"\n💓 Heart Rate Data:")
        print(f"   Data points: {len(heart_rate_df)}")
        print(f"   Duration: {len(heart_rate_df)}s ({len(heart_rate_df)/60:.2f} min)")
        print(f"   Range: {int(heart_rate_df['heart_rate'].min())}-{int(heart_rate_df['heart_rate'].max())} BPM")
        
        # Load heart icon (preserve original colors - no alteration)
        try:
            heart_icon_pil = Image.open(heart_icon_path).convert('RGBA')
            print(f"\n✅ Loaded heart icon: {heart_icon_path}")
        except Exception as e:
            print(f"\n❌ Could not load heart icon: {e}")
            return False
        
        # Check for font regular
        font_regular_path = font_path.parent / "Poppins-Regular.ttf"
        if not font_regular_path.exists():
            print(f"\n❌ Poppins-Regular.ttf not found: {font_regular_path}")
            print(f"   Please run: python src/download_assets.py")
            return False
        
        # Calculate scale factors based on reference dimensions
        # Reference: 1261x2242 (portrait 9:16)
        ref_width = 1261
        ref_height = 2242
        scale_x = width / ref_width
        scale_y = height / ref_height
        
        print(f"\n📐 Scale Factors:")
        print(f"   X Scale: {scale_x:.3f}")
        print(f"   Y Scale: {scale_y:.3f}")
        
        # Calculate positions and sizes based on exact specifications
        heart_left_percent = 266 / ref_width  # 21.1%
        heart_top_percent = 1692 / ref_height  # 75.5%
        heart_size_percent = 198 / ref_width  # 15.7%
        
        text_left_percent = 483 / ref_width  # 38.3%
        text_top_percent = 1713 / ref_height  # 76.4%
        font_size_percent = 120 / ref_width  # 9.5%
        
        # Apply to actual video dimensions
        heart_x = int(heart_left_percent * width)
        heart_y = int(heart_top_percent * height)
        heart_size = int(heart_size_percent * width)
        
        text_x = int(text_left_percent * width)
        text_y = int(text_top_percent * height)
        font_size = int(font_size_percent * width)
        
        # Stroke width scales with font size
        stroke_width = max(int(font_size * 0.067), 8)  # ~8-10px at reference size
        
        print(f"\n🎯 Overlay Positions:")
        print(f"   Heart: ({heart_x}, {heart_y}) size={heart_size}px")
        print(f"   Text: ({text_x}, {text_y}) font={font_size}px")
        print(f"   Stroke: {stroke_width}px")
        
        temp_frames_dir = Path(tempfile.mkdtemp(prefix="overlay_frames_"))
        
        print(f"\n⚙️  Processing Settings:")
        print(f"   Style: React Native Layout (Heart Icon + BPM Text + Chart)")
        print(f"   Bottom: Heart icon + text overlay")
        print(f"   Top: Animated heart rate chart (25% of screen)")
        print(f"   Chart: Progressive fill, smooth curves, gradient")
        print(f"   Output: {output_video}")
        print()
        
        chart_height = int(height * 0.25)
        chart_width = width
        
        with tqdm(total=total_frames, desc="Rendering", unit="frames", ncols=80) as pbar:
            for frame_count in range(total_frames):
                frame = np.zeros((height, width, 4), dtype=np.uint8)
                current_time = frame_count / fps
                
                # Get heart rate for current time
                heart_rate = get_heart_rate_at_time(heart_rate_df, current_time)
                
                # Calculate pulse scale (0.85 to 1.15)
                pulse_scale = calculate_pulse_scale(current_time, heart_rate)
                
                # Resize heart icon with pulse effect
                current_heart_size = int(heart_size * pulse_scale)
                heart_icon_scaled = heart_icon_pil.resize(
                    (current_heart_size, current_heart_size),
                    Image.Resampling.LANCZOS
                )
                # Convert PIL RGBA to numpy array (preserve RGB color channels)
                heart_icon_np = np.array(heart_icon_scaled)
                # PIL uses RGB, OpenCV uses BGR - but for RGBA overlay we keep RGB
                # The overlay_image_alpha function handles this correctly
                
                # Center the pulsing heart on the base position
                heart_x_centered = heart_x + (heart_size - current_heart_size) // 2
                heart_y_centered = heart_y + (heart_size - current_heart_size) // 2
                
                # Overlay heart icon (convert RGB to BGR to preserve original colors)
                frame = overlay_image_alpha(frame, heart_icon_np, heart_x_centered, heart_y_centered, convert_rgb_to_bgr=True)
                
                # Create text with stroke (number bold, "BPM" regular)
                text_img = create_text_with_stroke(
                    heart_rate,
                    font_path,
                    font_regular_path,
                    font_size,
                    text_color=(255, 255, 255),
                    stroke_color=(0, 0, 0),
                    stroke_width=stroke_width
                )
                
                # Overlay text (also convert RGB to BGR for PIL-generated text)
                frame = overlay_image_alpha(frame, text_img, text_x, text_y, convert_rgb_to_bgr=True)
                
                # Create and overlay heart rate chart at top of screen (optional, slow)
                if include_chart:
                    chart_img = create_heart_rate_chart(
                        heart_rate_df,
                        current_time,
                        chart_width,
                        chart_height,
                        duration
                    )
                    if chart_img.size > 0:
                        frame = overlay_image_alpha(frame, chart_img, 0, 0, convert_rgb_to_bgr=True)
                
                frame_path = temp_frames_dir / f"frame_{frame_count:05d}.png"
                cv2.imwrite(str(frame_path), frame)
                pbar.update(1)
        
        print(f"\n🎬 Encoding transparent WebM video...")
        if not encode_transparent_video(temp_frames_dir, output_video, fps):
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            return False
        
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        
        print(f"\n✅ Processing complete!")
        print(f"   Processed {total_frames} frames")
        print(f"   Output saved to: {output_video}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Professional Video Overlay System - Heart Rate Display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/overlay_video.py
  python src/overlay_video.py --output output/overlay.webm
  python src/overlay_video.py --csv input/mydata.csv

Design Specifications:
  Transparent canvas: 1080x1920 pixels (portrait 9:16)
  Heart icon: 21.1% from left, 75.5% from top, 15.7% width
  Text: Poppins-Bold, 9.5% width, white with black stroke
  Animation: ±15% pulse synced to actual BPM
  Output: WebM with alpha for compositing over other videos
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/transparent_overlay.webm',
        help='Output video file (default: output/transparent_overlay.webm)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Frames per second (default: 30)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=1080,
        help='Canvas width in pixels (default: 1080)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=1920,
        help='Canvas height in pixels (default: 1920)'
    )
    
    parser.add_argument(
        '--no-chart',
        action='store_true',
        help='Skip heart rate chart (faster, for testing)'
    )
    
    parser.add_argument(
        '--csv', '-c',
        type=str,
        default='input/heartrate.csv',
        help='Heart rate CSV file (default: input/heartrate.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 TRANSPARENT OVERLAY VIDEO SYSTEM")
    print("   Heart Rate Display - 1080x1920 Canvas")
    print("=" * 60)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    output_video = Path(args.output)
    csv_file = Path(args.csv)
    font_path = project_root / "assets" / "fonts" / "Poppins-Bold.ttf"
    heart_icon_path = project_root / "assets" / "images" / "heart.png"
    
    # Check input files
    if not csv_file.exists():
        print(f"\n❌ CSV file not found: {csv_file}")
        print(f"   Please add your heart rate data to: {csv_file}")
        return 1
    
    if not font_path.exists():
        print(f"\n❌ Font not found: {font_path}")
        print(f"   Please run: python src/download_assets.py")
        return 1
    
    if not heart_icon_path.exists():
        print(f"\n❌ Heart icon not found: {heart_icon_path}")
        print(f"   Please run: python src/download_assets.py")
        return 1
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file)
        
        # Validate columns
        if 'heart_rate' not in df.columns:
            print(f"\n❌ CSV must have 'heart_rate' column")
            print(f"   Found columns: {list(df.columns)}")
            return 1
        
        # Check for missing values
        if df['heart_rate'].isna().any():
            print(f"\n❌ CSV contains missing heart rate values")
            return 1
        
        print(f"\n✅ Loaded CSV: {csv_file}")
        
    except Exception as e:
        print(f"\n❌ Could not load CSV: {e}")
        return 1
    
    # Create output directory
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # Process video (transparent canvas, no input video)
    success = process_video(
        output_video,
        df,
        font_path,
        heart_icon_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        include_chart=not args.no_chart
    )
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS!")
        print("=" * 60)
        print(f"\n📹 Your transparent overlay is ready: {output_video}")
        print(f"   Use this WebM file as an overlay layer in your video editor.")
        print()
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ FAILED")
        print("=" * 60)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

