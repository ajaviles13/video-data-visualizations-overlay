#!/usr/bin/env python3
"""
Professional Video Overlay System - Heart Rate & Temperature Display
Matches exact Figma/React Native design specifications with dual charts
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, CubicSpline


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
    Overlay RGBA image with transparency onto BGR video frame.
    
    Args:
        background: BGR background image (video frame)
        overlay: RGBA overlay image (text or icon)
        x: X position (left edge)
        y: Y position (top edge)
        convert_rgb_to_bgr: If True, convert RGB channels to BGR (for PIL images)
        
    Returns:
        Combined BGR image
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
        # Adjust overlay if it extends beyond boundaries
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(background.shape[1], x + w)
        y_end = min(background.shape[0], y + h)
        
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
    
    # Extract alpha channel (0-255)
    alpha = overlay[:, :, 3] / 255.0
    
    # Get region of interest
    roi = background[y:y+h, x:x+w]
    
    # Blend each color channel
    for c in range(3):
        roi[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c])
    
    background[y:y+h, x:x+w] = roi
    
    return background


def interpolate_temperature_data(temp_df: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """
    Interpolate temperature data to match heart rate data length using Cubic Spline.
    
    Temperature data is recorded every 5 seconds, but we need per-second data
    to match the heart rate data frequency.
    
    Args:
        temp_df: DataFrame with temperature data (timestamp, temp_data)
        target_length: Target number of data points (seconds) to match HR data
        
    Returns:
        DataFrame with interpolated temperature data at 1-second intervals
    """
    if len(temp_df) < 2:
        print("⚠️  Warning: Not enough temperature data for interpolation")
        return temp_df
    
    # Original timestamps and temperature values
    original_timestamps = temp_df['timestamp'].values
    original_temps = temp_df['temp_data'].values
    
    # Create cubic spline interpolator
    cs = CubicSpline(original_timestamps, original_temps)
    
    # Generate new timestamps from 0 to target_length-1 (per second)
    new_timestamps = np.arange(0, target_length)
    
    # Interpolate temperature values
    interpolated_temps = cs(new_timestamps)
    
    # Create new DataFrame
    interpolated_df = pd.DataFrame({
        'timestamp': new_timestamps,
        'temp_data': interpolated_temps
    })
    
    print(f"   Original temp data: {len(temp_df)} points (every 5s)")
    print(f"   Interpolated to: {len(interpolated_df)} points (every 1s)")
    print(f"   Temp range: {interpolated_temps.min():.1f}°F - {interpolated_temps.max():.1f}°F")
    
    return interpolated_df


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
        chart_height: Height of chart in pixels (15% of video height)
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
    
    # Plot the line (red/pink for heart rate)
    line_color = '#FF6E65'
    ax.plot(x_scaled, y_smooth, color=line_color, linewidth=2.5, zorder=3)
    
    # Fill area under the curve with gradient effect
    fill_color = '#FF6E65'
    ax.fill_between(x_scaled, hr_min_display, y_smooth, 
                     color=fill_color, alpha=0.6, zorder=2)
    
    # Set axis limits
    ax.set_xlim(0, video_duration)
    ax.set_ylim(hr_min_display, hr_max_display)
    
    # Remove all spines
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


def create_temperature_chart(
    temp_df: pd.DataFrame,
    current_time: float,
    chart_width: int,
    chart_height: int,
    video_duration: float
) -> np.ndarray:
    """
    Create a progressive temperature chart that fills from left to right.
    
    Args:
        temp_df: DataFrame with temperature data (interpolated to per-second)
        current_time: Current video time in seconds
        chart_width: Width of chart in pixels
        chart_height: Height of chart in pixels (15% of video height)
        video_duration: Total video duration in seconds
        
    Returns:
        RGBA numpy array with chart overlay
    """
    # Get data up to current time
    max_time_idx = int(current_time)
    if max_time_idx < 1:
        max_time_idx = 1
    
    # Get temperature data up to current time
    data_slice = temp_df.iloc[:max_time_idx + 1].copy()
    
    if len(data_slice) < 2:
        # Not enough data yet, return transparent image
        return np.zeros((chart_height, chart_width, 4), dtype=np.uint8)
    
    # Get time points (0 to current_time)
    if 'timestamp' in data_slice.columns:
        time_points = data_slice['timestamp'].values
    else:
        time_points = np.arange(len(data_slice))
    
    temp_values = data_slice['temp_data'].values
    
    # Get min and max for auto-scaling
    temp_min = temp_df['temp_data'].min()
    temp_max = temp_df['temp_data'].max()
    
    # Add some padding to the range
    temp_range = temp_max - temp_min
    temp_min_display = temp_min - temp_range * 0.05
    temp_max_display = temp_max + temp_range * 0.05
    
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
    
    # Create smooth curve using spline interpolation
    if len(time_points) > 3:
        # Use spline for smooth curve
        x_smooth = np.linspace(time_points[0], time_points[-1], len(time_points) * 10)
        spl = make_interp_spline(time_points, temp_values, k=min(3, len(time_points) - 1))
        y_smooth = spl(x_smooth)
    else:
        x_smooth = time_points
        y_smooth = temp_values
    
    # Scale x-axis to match video duration (0 to video_duration)
    x_scaled = (x_smooth / video_duration) * video_duration
    
    # Plot the line (blue for cold temperature)
    line_color = '#5DADE2'
    ax.plot(x_scaled, y_smooth, color=line_color, linewidth=2.5, zorder=3)
    
    # Fill area under the curve with gradient effect
    fill_color = '#5DADE2'
    ax.fill_between(x_scaled, temp_min_display, y_smooth, 
                     color=fill_color, alpha=0.6, zorder=2)
    
    # Set axis limits
    ax.set_xlim(0, video_duration)
    ax.set_ylim(temp_min_display, temp_max_display)
    
    # Remove all spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Set Y-axis ticks to show only min and max
    ax.set_yticks([temp_min, temp_max])
    ax.set_yticklabels([f'{int(temp_min)}', f'{int(temp_max)}'], 
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


def get_temperature_at_time(df: pd.DataFrame, time_seconds: float) -> float:
    """
    Get temperature value for a specific time.
    
    Args:
        df: DataFrame with temperature data
        time_seconds: Time in seconds
        
    Returns:
        Temperature value (interpolated if needed)
    """
    if 'timestamp' in df.columns:
        # Find closest timestamp
        idx = df['timestamp'].searchsorted(time_seconds)
        if idx >= len(df):
            idx = len(df) - 1
        elif idx > 0 and abs(df.iloc[idx-1]['timestamp'] - time_seconds) < abs(df.iloc[idx]['timestamp'] - time_seconds):
            idx = idx - 1
        return float(df.iloc[idx]['temp_data'])
    else:
        # Use row index as timestamp (1 row = 1 second)
        idx = int(time_seconds)
        if idx >= len(df):
            idx = len(df) - 1
        return float(df.iloc[idx]['temp_data'])


def create_temperature_text_with_stroke(
    temperature: float,
    font_bold_path: Path,
    font_regular_path: Path,
    font_size: int,
    text_color: tuple = (255, 255, 255),
    stroke_color: tuple = (0, 0, 0),
    stroke_width: int = 8
) -> np.ndarray:
    """
    Create temperature text image with black stroke using PIL.
    Temperature value is bold, "ºF" is regular weight.
    
    Args:
        temperature: Temperature value (e.g., 79.9)
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
    
    # Split text into number (bold) and " ºF" (regular)
    number_text = f"{temperature:.1f}"
    unit_text = " ºF"
    
    # Get bounding boxes for both parts
    dummy_img = Image.new('RGBA', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    number_bbox = dummy_draw.textbbox((0, 0), number_text, font=font_bold)
    number_width = number_bbox[2] - number_bbox[0]
    number_height = number_bbox[3] - number_bbox[1]
    
    unit_bbox = dummy_draw.textbbox((0, 0), unit_text, font=font_regular)
    unit_width = unit_bbox[2] - unit_bbox[0]
    unit_height = unit_bbox[3] - unit_bbox[1]
    
    # Total dimensions
    total_width = number_width + unit_width
    total_height = max(number_height, unit_height)
    
    # Create image with padding for stroke
    padding = stroke_width * 2
    img_width = total_width + padding * 2
    img_height = total_height + padding * 2
    
    img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Calculate positions
    number_x = padding - number_bbox[0]
    number_y = padding - number_bbox[1]
    
    unit_x = number_x + number_width
    unit_y = padding - unit_bbox[1]
    
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
        draw.text((unit_x + dx, unit_y + dy), unit_text, font=font_regular, fill=stroke_color)
    
    # Draw main text on top (white)
    draw.text((number_x, number_y), number_text, font=font_bold, fill=text_color)
    draw.text((unit_x, unit_y), unit_text, font=font_regular, fill=text_color)
    
    # Convert to numpy array
    return np.array(img)


def process_video(
    input_video: Path,
    output_video: Path,
    heart_rate_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    font_path: Path,
    heart_icon_path: Path,
    temp_icon_path: Path
) -> bool:
    """
    Process video and add heart rate + temperature overlay with dual charts.
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        heart_rate_df: DataFrame with heart rate data
        temp_df: DataFrame with temperature data
        font_path: Path to Poppins-Bold.ttf
        heart_icon_path: Path to heart.png
        temp_icon_path: Path to TempIcon.png
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open video
        cap = cv2.VideoCapture(str(input_video))
        
        if not cap.isOpened():
            print(f"❌ Could not open video file: {input_video}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\n📹 Video Information:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {duration:.2f}s ({duration/60:.2f} min)")
        
        print(f"\n💓 Heart Rate Data:")
        print(f"   Data points: {len(heart_rate_df)}")
        print(f"   Duration: {len(heart_rate_df)}s ({len(heart_rate_df)/60:.2f} min)")
        print(f"   Range: {int(heart_rate_df['heart_rate'].min())}-{int(heart_rate_df['heart_rate'].max())} BPM")
        
        print(f"\n🌡️  Temperature Data (pre-interpolation):")
        print(f"   Data points: {len(temp_df)}")
        print(f"   Duration: {temp_df['timestamp'].max():.0f}s")
        
        # Interpolate temperature data to match heart rate data length
        print(f"\n🔄 Interpolating temperature data...")
        target_length = len(heart_rate_df)
        temp_df_interpolated = interpolate_temperature_data(temp_df, target_length)
        
        # Load heart icon (preserve original colors - no alteration, no pulsing)
        try:
            heart_icon_pil = Image.open(heart_icon_path).convert('RGBA')
            print(f"\n✅ Loaded heart icon: {heart_icon_path}")
        except Exception as e:
            print(f"\n❌ Could not load heart icon: {e}")
            return False
        
        # Load temperature icon
        try:
            temp_icon_pil = Image.open(temp_icon_path).convert('RGBA')
            print(f"✅ Loaded temperature icon: {temp_icon_path}")
        except Exception as e:
            print(f"\n❌ Could not load temperature icon: {e}")
            return False
        
        # Check for font regular
        font_regular_path = font_path.parent / "Poppins-Regular.ttf"
        if not font_regular_path.exists():
            print(f"\n❌ Poppins-Regular.ttf not found: {font_regular_path}")
            print(f"   Please run: python src/download_assets.py")
            return False
        
        # Calculate scale factors based on reference dimensions
        ref_width = 1261
        ref_height = 2242
        scale_x = width / ref_width
        scale_y = height / ref_height
        
        print(f"\n📐 Scale Factors:")
        print(f"   X Scale: {scale_x:.3f}")
        print(f"   Y Scale: {scale_y:.3f}")
        
        # Calculate positions and sizes for DUAL DISPLAY layout (heart + temperature)
        # Reference dimensions for icon size and font
        heart_size_percent = 198 / ref_width  # 15.7% (icon size from reference)
        font_size_percent = 120 / ref_width  # 9.5% (font size from reference)
        
        # Calculate icon and font sizes (reduced to 85% to fit on screen)
        icon_size = int(heart_size_percent * width * 0.85)  # Reduced from 100% to 85%
        font_size = int(font_size_percent * width * 0.90)  # Slightly reduced font
        temp_icon_size = icon_size  # Same size as heart icon
        
        # Stroke width scales with font size
        stroke_width = max(int(font_size * 0.067), 8)
        
        # Calculate fixed text widths to prevent bouncing (using max possible values)
        # Pre-calculate maximum text widths for anchored positioning
        dummy_font_bold = ImageFont.truetype(str(font_path), font_size)
        dummy_font_regular = ImageFont.truetype(str(font_regular_path), font_size)
        dummy_img = Image.new('RGBA', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        # Max heart rate text: "999 BPM"
        max_hr_bbox = dummy_draw.textbbox((0, 0), "999", font=dummy_font_bold)
        max_hr_num_width = max_hr_bbox[2] - max_hr_bbox[0]
        bpm_bbox = dummy_draw.textbbox((0, 0), " BPM", font=dummy_font_regular)
        bpm_width = bpm_bbox[2] - bpm_bbox[0]
        max_hr_text_width = max_hr_num_width + bpm_width + stroke_width * 4
        
        # Max temperature text: "999.9 ºF"
        max_temp_bbox = dummy_draw.textbbox((0, 0), "999.9", font=dummy_font_bold)
        max_temp_num_width = max_temp_bbox[2] - max_temp_bbox[0]
        temp_unit_bbox = dummy_draw.textbbox((0, 0), " ºF", font=dummy_font_regular)
        temp_unit_width = temp_unit_bbox[2] - temp_unit_bbox[0]
        max_temp_text_width = max_temp_num_width + temp_unit_width + stroke_width * 4
        
        # Calculate layout dimensions with optimized spacing
        hr_icon_text_gap = int(width * 0.01)  # 2% gap between heart icon and text
        temp_icon_text_gap = int(width * 0)  # 0% gap for temp (icon has built-in clearance)
        container_gap = int(width * 0)  # 0% gap between containers (reduced from 1%)
        edge_padding = int(width * 0.02)  # 2% padding from screen edges
        
        # Heart container: icon + gap + text
        hr_container_width = icon_size + hr_icon_text_gap + max_hr_text_width
        # Temperature container: icon + gap + text
        temp_container_width = temp_icon_size + temp_icon_text_gap + max_temp_text_width
        
        # Total width of both containers with gap
        total_layout_width = hr_container_width + container_gap + temp_container_width
        
        # Center the entire layout horizontally with edge padding
        layout_start_x = max(edge_padding, (width - total_layout_width) // 2)
        
        # Vertical position (75.5% from top as per original spec)
        base_y_percent = 1692 / ref_height  # 75.5%
        base_y = int(base_y_percent * height)
        
        # Heart container positions
        hr_icon_x = layout_start_x
        hr_icon_y = base_y
        hr_text_x = hr_icon_x + icon_size + hr_icon_text_gap
        hr_text_y = base_y + int((icon_size - font_size) * 0.5)  # Vertically center text with icon
        
        # Temperature container positions (to the right of heart container)
        temp_container_start = layout_start_x + hr_container_width + container_gap
        temp_icon_x = temp_container_start
        temp_icon_y = base_y
        temp_text_x = temp_icon_x + temp_icon_size + temp_icon_text_gap
        temp_text_y = base_y + int((temp_icon_size - font_size) * 0.5)  # Vertically center text with icon
        
        print(f"\n🎯 Overlay Positions (Dual Display Layout - Optimized):")
        print(f"   Icon size: {icon_size}px (85% of reference)")
        print(f"   Font size: {font_size}px (90% of reference)")
        print(f"   Stroke: {stroke_width}px")
        print(f"   Heart icon: ({hr_icon_x}, {hr_icon_y})")
        print(f"   Heart text: ({hr_text_x}, {hr_text_y}) [gap: {hr_icon_text_gap}px, max width: {max_hr_text_width}px]")
        print(f"   Temp icon: ({temp_icon_x}, {temp_icon_y})")
        print(f"   Temp text: ({temp_text_x}, {temp_text_y}) [gap: {temp_icon_text_gap}px, max width: {max_temp_text_width}px]")
        print(f"   Container gap: {container_gap}px (reduced from 8% to 4%)")
        print(f"   Edge padding: {edge_padding}px")
        print(f"   Total layout width: {total_layout_width}px (centered with safety padding)")
        
        # Setup video writer with H.264 codec
        codecs = ['avc1', 'H264', 'X264', 'mp4v']
        out = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            if out.isOpened():
                print(f"\n✅ Using codec: {codec}")
                break
            else:
                out.release()
        
        if out is None or not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Could not create output video file: {output_video}")
            return False
        
        print(f"\n⚙️  Processing Settings:")
        print(f"   Style: Dual Chart Layout (HR + Temperature)")
        print(f"   Top 0-15%: Heart rate chart (red)")
        print(f"   Top 15-30%: Temperature chart (blue)")
        print(f"   Bottom: Static heart icon + text overlay")
        print(f"   Output: {output_video}")
        print()
        
        frame_count = 0
        
        # Calculate chart dimensions (15% each, stacked)
        hr_chart_height = int(height * 0.15)
        temp_chart_height = int(height * 0.15)
        chart_width = width
        
        # Resize icons ONCE (static size, no animations)
        heart_icon_scaled = heart_icon_pil.resize(
            (icon_size, icon_size),
            Image.Resampling.LANCZOS
        )
        heart_icon_np = np.array(heart_icon_scaled)
        
        temp_icon_scaled = temp_icon_pil.resize(
            (temp_icon_size, temp_icon_size),
            Image.Resampling.LANCZOS
        )
        temp_icon_np = np.array(temp_icon_scaled)
        
        # Process each frame with progress bar
        with tqdm(total=total_frames, desc="Rendering", unit="frames", ncols=80) as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Calculate current time in seconds
                current_time = frame_count / fps
                
                # Get heart rate and temperature for current time
                heart_rate = get_heart_rate_at_time(heart_rate_df, current_time)
                temperature = get_temperature_at_time(temp_df_interpolated, current_time)
                
                # === HEART RATE DISPLAY (LEFT) ===
                # Overlay STATIC heart icon
                frame = overlay_image_alpha(frame, heart_icon_np, hr_icon_x, hr_icon_y, convert_rgb_to_bgr=True)
                
                # Create and overlay heart rate text
                hr_text_img = create_text_with_stroke(
                    heart_rate,
                    font_path,
                    font_regular_path,
                    font_size,
                    text_color=(255, 255, 255),
                    stroke_color=(0, 0, 0),
                    stroke_width=stroke_width
                )
                frame = overlay_image_alpha(frame, hr_text_img, hr_text_x, hr_text_y, convert_rgb_to_bgr=True)
                
                # === TEMPERATURE DISPLAY (RIGHT) ===
                # Overlay STATIC temperature icon
                frame = overlay_image_alpha(frame, temp_icon_np, temp_icon_x, temp_icon_y, convert_rgb_to_bgr=True)
                
                # Create and overlay temperature text
                temp_text_img = create_temperature_text_with_stroke(
                    temperature,
                    font_path,
                    font_regular_path,
                    font_size,
                    text_color=(255, 255, 255),
                    stroke_color=(0, 0, 0),
                    stroke_width=stroke_width
                )
                frame = overlay_image_alpha(frame, temp_text_img, temp_text_x, temp_text_y, convert_rgb_to_bgr=True)
                
                # Create and overlay heart rate chart at top (0-15%)
                hr_chart_img = create_heart_rate_chart(
                    heart_rate_df,
                    current_time,
                    chart_width,
                    hr_chart_height,
                    duration
                )
                
                if hr_chart_img.size > 0:
                    frame = overlay_image_alpha(frame, hr_chart_img, 0, 0, convert_rgb_to_bgr=True)
                
                # Create and overlay temperature chart below HR chart (15-30%)
                temp_chart_img = create_temperature_chart(
                    temp_df_interpolated,
                    current_time,
                    chart_width,
                    temp_chart_height,
                    duration
                )
                
                if temp_chart_img.size > 0:
                    temp_chart_y = hr_chart_height  # Start right after HR chart
                    frame = overlay_image_alpha(frame, temp_chart_img, 0, temp_chart_y, convert_rgb_to_bgr=True)
                
                # Write frame
                out.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"\n✅ Processing complete!")
        print(f"   Processed {frame_count} frames")
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
        description="Professional Video Overlay System - Heart Rate & Temperature Display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/overlay_video_hr_and_temp_data.py
  python src/overlay_video_hr_and_temp_data.py --output output/result.mp4

Features:
  - Dual chart display (Heart Rate + Temperature)
  - Heart rate chart: 0-15% (red/pink)
  - Temperature chart: 15-30% (blue)
  - Cubic spline interpolation for temperature data
  - Static heart icon (no pulsing animation)
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='input/video.mp4',
        help='Input video file (default: input/video.mp4)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/video_with_hr_and_temp.mp4',
        help='Output video file (default: output/video_with_hr_and_temp.mp4)'
    )
    
    parser.add_argument(
        '--hr-csv',
        type=str,
        default='input/heartrate.csv',
        help='Heart rate CSV file (default: input/heartrate.csv)'
    )
    
    parser.add_argument(
        '--temp-csv',
        type=str,
        default='input/temperature_data.csv',
        help='Temperature CSV file (default: input/temperature_data.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 PROFESSIONAL VIDEO OVERLAY SYSTEM")
    print("   Heart Rate & Temperature Display - Dual Charts")
    print("=" * 60)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    input_video = Path(args.input)
    output_video = Path(args.output)
    hr_csv_file = Path(args.hr_csv)
    temp_csv_file = Path(args.temp_csv)
    font_path = project_root / "assets" / "fonts" / "Poppins-Bold.ttf"
    heart_icon_path = project_root / "assets" / "images" / "heart.png"
    temp_icon_path = project_root / "assets" / "images" / "TempIcon.png"
    
    # Check input files
    if not input_video.exists():
        print(f"\n❌ Input video not found: {input_video}")
        return 1
    
    if not hr_csv_file.exists():
        print(f"\n❌ Heart rate CSV not found: {hr_csv_file}")
        return 1
    
    if not temp_csv_file.exists():
        print(f"\n❌ Temperature CSV not found: {temp_csv_file}")
        return 1
    
    if not font_path.exists():
        print(f"\n❌ Font not found: {font_path}")
        print(f"   Please run: python src/download_assets.py")
        return 1
    
    if not heart_icon_path.exists():
        print(f"\n❌ Heart icon not found: {heart_icon_path}")
        print(f"   Please run: python src/download_assets.py")
        return 1
    
    if not temp_icon_path.exists():
        print(f"\n❌ Temperature icon not found: {temp_icon_path}")
        print(f"   Please ensure TempIcon.png exists in assets/images/")
        return 1
    
    # Load Heart Rate CSV
    try:
        hr_df = pd.read_csv(hr_csv_file)
        
        if 'heart_rate' not in hr_df.columns:
            print(f"\n❌ Heart rate CSV must have 'heart_rate' column")
            return 1
        
        if hr_df['heart_rate'].isna().any():
            print(f"\n❌ Heart rate CSV contains missing values")
            return 1
        
        print(f"\n✅ Loaded heart rate CSV: {hr_csv_file}")
        
    except Exception as e:
        print(f"\n❌ Could not load heart rate CSV: {e}")
        return 1
    
    # Load Temperature CSV
    try:
        temp_df = pd.read_csv(temp_csv_file)
        
        if 'temp_data' not in temp_df.columns or 'timestamp' not in temp_df.columns:
            print(f"\n❌ Temperature CSV must have 'timestamp' and 'temp_data' columns")
            return 1
        
        if temp_df['temp_data'].isna().any():
            print(f"\n❌ Temperature CSV contains missing values")
            return 1
        
        print(f"✅ Loaded temperature CSV: {temp_csv_file}")
        
    except Exception as e:
        print(f"\n❌ Could not load temperature CSV: {e}")
        return 1
    
    # Create output directory
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # Process video
    success = process_video(
        input_video,
        output_video,
        hr_df,
        temp_df,
        font_path,
        heart_icon_path,
        temp_icon_path
    )
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS!")
        print("=" * 60)
        print(f"\n📹 Your video is ready: {output_video}")
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

