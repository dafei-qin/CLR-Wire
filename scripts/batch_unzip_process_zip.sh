#!/bin/bash

# Script to process .7z files in /root/CAD/data/abc/
# Extracts abc_XXXX_step_v00.7z to Y/XXXX where Y is the second digit of XXXX
# Then deletes the .7z file

SOURCE_DIR="/data/ssd/CAD/data/abc/9"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory $SOURCE_DIR does not exist"
    exit 1
fi

# Check if 7z command is available
if ! command -v 7z &> /dev/null; then
    echo "Error: 7z command not found. Please install p7zip-full"
    exit 1
fi

cd "$SOURCE_DIR" || exit 1

# Find all .7z files and process them sequentially
find . -maxdepth 1 -name "*.7z" -type f | sort | while read -r archive; do
    # Get just the filename without path
    filename=$(basename "$archive")
    
    # Extract the 4-digit number from filename (e.g., abc_0050_step_v00.7z -> 0050)
    if [[ $filename =~ abc_([0-9]{4})_step_v00\.7z ]]; then
        full_num="${BASH_REMATCH[1]}"
        
        # Get the second digit (e.g., 0050 -> 5, 0077 -> 7)
        second_digit="${full_num:1:1}"
        
        # Create target directory structure
        target_dir="${second_digit}/${full_num}"
        
        echo "Processing: $filename"
        echo "  Target directory: $target_dir"
        
        # Create directory if it doesn't exist
        mkdir -p "$target_dir"
        
        # Extract the archive to the target directory
        if 7z x "$filename" -o"$target_dir" -y; then
            echo "  ✓ Extracted successfully"
            
            # Delete the .7z file
            rm "$filename"
            echo "  ✓ Deleted $filename"
        else
            echo "  ✗ Error extracting $filename"
        fi
        
        echo ""
    else
        echo "Warning: $filename does not match expected pattern (abc_XXXX_step_v00.7z)"
    fi
done

echo "Processing complete!"

