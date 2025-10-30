#!/usr/bin/env python3
"""
CSV Validation Script for Heart Rate Data
Validates format, checks values, and displays statistics
"""

import sys
import pandas as pd
import argparse
from pathlib import Path


def validate_csv(csv_path: Path, expected_duration: float = None) -> bool:
    """
    Validate heart rate CSV file.
    
    Args:
        csv_path: Path to CSV file
        expected_duration: Expected video duration in seconds (optional)
        
    Returns:
        True if valid, False otherwise
    """
    print("=" * 60)
    print("📊 HEART RATE CSV VALIDATION")
    print("=" * 60)
    print()
    
    # Check file exists
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return False
    
    print(f"📁 File: {csv_path}")
    print(f"   Size: {csv_path.stat().st_size:,} bytes")
    print()
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSV loaded successfully")
    except Exception as e:
        print(f"❌ Could not load CSV: {e}")
        return False
    
    # Check columns
    print(f"\n📋 Columns Found:")
    for col in df.columns:
        print(f"   • {col}")
    
    if 'heart_rate' not in df.columns:
        print(f"\n❌ ERROR: Missing required column 'heart_rate'")
        print(f"   Your CSV must have at least a 'heart_rate' column")
        return False
    
    print(f"\n✓ Required column 'heart_rate' found")
    
    has_timestamp = 'timestamp' in df.columns
    if has_timestamp:
        print(f"✓ Optional column 'timestamp' found")
    else:
        print(f"ℹ️  No 'timestamp' column (will use row index as seconds)")
    
    # Check for missing values
    missing_hr = df['heart_rate'].isna().sum()
    if missing_hr > 0:
        print(f"\n❌ ERROR: Found {missing_hr} missing heart rate values")
        print(f"   Please fill in all heart rate values")
        return False
    
    print(f"\n✓ No missing values")
    
    # Check row count
    row_count = len(df)
    duration_seconds = row_count
    duration_minutes = duration_seconds / 60.0
    
    print(f"\n📊 Data Summary:")
    print(f"   Total Rows: {row_count}")
    print(f"   Duration: {duration_seconds}s ({duration_minutes:.2f} min)")
    
    if expected_duration:
        diff = abs(duration_seconds - expected_duration)
        if diff > 5:
            print(f"   ⚠️  Warning: Expected {expected_duration:.1f}s, got {duration_seconds}s")
            print(f"   Difference: {diff:.1f}s")
        else:
            print(f"   ✓ Matches expected duration ({expected_duration:.1f}s)")
    
    # Check heart rate values
    hr_values = df['heart_rate']
    hr_min = hr_values.min()
    hr_max = hr_values.max()
    hr_mean = hr_values.mean()
    hr_median = hr_values.median()
    
    print(f"\n💓 Heart Rate Statistics:")
    print(f"   Minimum: {int(hr_min)} BPM")
    print(f"   Maximum: {int(hr_max)} BPM")
    print(f"   Average: {int(hr_mean)} BPM")
    print(f"   Median: {int(hr_median)} BPM")
    
    # Validate realistic range
    if hr_min < 40 or hr_max > 220:
        print(f"\n⚠️  WARNING: Heart rate values outside typical range (40-220 BPM)")
        if hr_min < 40:
            print(f"   Minimum value {int(hr_min)} BPM is unusually low")
        if hr_max > 220:
            print(f"   Maximum value {int(hr_max)} BPM is unusually high")
        print(f"   Please verify your data")
        
        response = input(f"\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    else:
        print(f"   ✓ Values within realistic range (40-220 BPM)")
    
    # Show data preview
    print(f"\n📄 Data Preview (First 5 Rows):")
    print(df.head().to_string(index=False))
    
    if row_count > 5:
        print(f"\n📄 Data Preview (Last 5 Rows):")
        print(df.tail().to_string(index=False))
    
    # Distribution analysis
    print(f"\n📊 Heart Rate Distribution:")
    ranges = [
        (40, 60, "Resting"),
        (60, 100, "Normal"),
        (100, 140, "Elevated"),
        (140, 180, "High"),
        (180, 220, "Maximum")
    ]
    
    for min_val, max_val, label in ranges:
        count = ((hr_values >= min_val) & (hr_values < max_val)).sum()
        if count > 0:
            percentage = (count / row_count) * 100
            print(f"   {label:12} ({min_val:3}-{max_val:3} BPM): {count:4} rows ({percentage:5.1f}%)")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("✅ CSV VALIDATION PASSED")
    print("=" * 60)
    print()
    print("Your CSV is ready to use!")
    print()
    print("Next steps:")
    print("  1. Add your video to input/video.mp4")
    print("  2. Run: python src/overlay_video.py")
    print()
    
    return True


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Validate heart rate CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/validate_csv.py
  python src/validate_csv.py --csv input/heartrate.csv
  python src/validate_csv.py --csv data.csv --duration 210

CSV Format:
  Required columns: heart_rate
  Optional columns: timestamp
  
  Example 1 (with timestamp):
    timestamp,heart_rate
    0,72
    1,75
    2,78
  
  Example 2 (without timestamp):
    heart_rate
    72
    75
    78
        """
    )
    
    parser.add_argument(
        '--csv', '-c',
        type=str,
        default='input/heartrate.csv',
        help='Path to CSV file (default: input/heartrate.csv)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help='Expected video duration in seconds (optional)'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    
    if validate_csv(csv_path, args.duration):
        return 0
    else:
        print("=" * 60)
        print("❌ VALIDATION FAILED")
        print("=" * 60)
        print()
        print("Please fix the errors above and try again.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
