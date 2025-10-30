#!/usr/bin/env python3
"""
Asset Downloader for Video Overlay System
Downloads required fonts and creates heart icon
"""

import sys
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import time


def download_file(url: str, destination: Path, max_retries: int = 3) -> bool:
    """
    Download file from URL with retry logic.
    
    Args:
        url: URL to download from
        destination: Path to save file
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading {destination.name}... (Attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Successfully downloaded {destination.name}")
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    return False


def create_heart_icon(size: int, output_path: Path, color: tuple = (255, 107, 107)) -> bool:
    """
    Create a red heart icon PNG with transparency.
    
    Args:
        size: Size of the icon (width and height)
        output_path: Path to save the PNG
        color: RGB color tuple for the heart
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🎨 Creating heart icon ({size}x{size} pixels)...")
        
        # Create transparent image
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Calculate heart shape coordinates
        center_x = size // 2
        center_y = int(size * 0.35)
        
        # Heart proportions
        radius = int(size * 0.28)
        left_center = (center_x - int(size * 0.22), center_y)
        right_center = (center_x + int(size * 0.22), center_y)
        
        # Draw left lobe
        draw.ellipse(
            [left_center[0] - radius, left_center[1] - radius,
             left_center[0] + radius, left_center[1] + radius],
            fill=color
        )
        
        # Draw right lobe
        draw.ellipse(
            [right_center[0] - radius, right_center[1] - radius,
             right_center[0] + radius, right_center[1] + radius],
            fill=color
        )
        
        # Draw bottom triangle
        bottom_point = (center_x, center_y + int(size * 0.68))
        left_point = (center_x - int(size * 0.48), center_y + int(radius * 0.3))
        right_point = (center_x + int(size * 0.48), center_y + int(radius * 0.3))
        
        draw.polygon([left_point, right_point, bottom_point], fill=color)
        
        # Fill center rectangle to smooth connection
        rect_top_left = (center_x - int(size * 0.22), center_y - int(radius * 0.3))
        rect_bottom_right = (center_x + int(size * 0.22), center_y + int(radius * 0.5))
        draw.rectangle([rect_top_left, rect_bottom_right], fill=color)
        
        # Save PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, 'PNG')
        
        print(f"✅ Heart icon created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create heart icon: {e}")
        return False


def validate_file(file_path: Path, min_size: int = 1000) -> bool:
    """
    Validate that a file exists and has reasonable size.
    
    Args:
        file_path: Path to file
        min_size: Minimum expected file size in bytes
        
    Returns:
        True if valid, False otherwise
    """
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    size = file_path.stat().st_size
    if size < min_size:
        print(f"❌ File too small ({size} bytes): {file_path}")
        return False
    
    print(f"✅ Validated {file_path.name} ({size:,} bytes)")
    return True


def main():
    """Main asset download function."""
    
    print("=" * 60)
    print("🎨 VIDEO OVERLAY SYSTEM - ASSET DOWNLOADER")
    print("=" * 60)
    print()
    
    # Define paths
    project_root = Path(__file__).parent.parent
    fonts_dir = project_root / "assets" / "fonts"
    images_dir = project_root / "assets" / "images"
    
    font_path = fonts_dir / "Poppins-Bold.ttf"
    heart_path = images_dir / "heart.png"
    
    # Track success
    success_count = 0
    total_count = 3
    
    # Download Poppins-Bold font
    print("📝 STEP 1: Download Poppins-Bold Font")
    print("-" * 60)
    
    font_url = "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf"
    
    if font_path.exists():
        print(f"ℹ️  Font already exists: {font_path}")
        if validate_file(font_path):
            success_count += 1
    else:
        if download_file(font_url, font_path):
            if validate_file(font_path):
                success_count += 1
    
    print()
    
    # Download Poppins-Regular font
    print("📝 STEP 2: Download Poppins-Regular Font")
    print("-" * 60)
    
    font_regular_path = fonts_dir / "Poppins-Regular.ttf"
    font_regular_url = "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf"
    
    if font_regular_path.exists():
        print(f"ℹ️  Font already exists: {font_regular_path}")
        if validate_file(font_regular_path):
            success_count += 1
    else:
        if download_file(font_regular_url, font_regular_path):
            if validate_file(font_regular_path):
                success_count += 1
    
    print()
    
    # Create heart icon
    print("❤️  STEP 3: Create Heart Icon")
    print("-" * 60)
    
    if heart_path.exists():
        print(f"ℹ️  Heart icon already exists: {heart_path}")
        if validate_file(heart_path, min_size=500):
            success_count += 1
    else:
        if create_heart_icon(198, heart_path, color=(255, 107, 107)):
            if validate_file(heart_path, min_size=500):
                success_count += 1
    
    print()
    print("=" * 60)
    
    # Final report
    if success_count == total_count:
        print("🎉 SUCCESS! All assets ready")
        print("=" * 60)
        print()
        print("📁 Asset Locations:")
        print(f"   Font (Bold): {font_path}")
        print(f"   Font (Regular): {fonts_dir / 'Poppins-Regular.ttf'}")
        print(f"   Heart: {heart_path}")
        print()
        print("🚀 Next Steps:")
        print("   1. Add your video to input/video.mp4")
        print("   2. Add your CSV to input/heartrate.csv")
        print("   3. Run: python src/overlay_video.py")
        print()
        return 0
    else:
        print(f"⚠️  WARNING: {success_count}/{total_count} assets ready")
        print("=" * 60)
        print()
        print("Some assets may be missing. Please check the errors above.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

