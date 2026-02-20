import whisper
import os
import glob
from pathlib import Path
from utils.log_utils import log_processing_file

# Load the Whisper model once
model = whisper.load_model("small")

def extract_text_from_videos(video_pattern:str, output_dir:str):
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)      
    
    # Find all .mp4 files in the input directory
    mp4_files = glob.glob(video_pattern)

    if not mp4_files:
        print("No .mp4 files found in the specified directory.")
    else:
        print(f"Found {len(mp4_files)} .mp4 files to process:")
        
        for video_file in mp4_files:
            log_processing_file(file=video_file)
            
            try:
                # Transcribe the video file
                result = model.transcribe(video_file, fp16=False, language="en")
                
                # Generate output filename based on input filename
                video_name = Path(video_file).stem
                output_filename = f"{video_name}_transcript.txt"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the transcript
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                
                print(f"✓ Transcript saved: {output_path}")
                
            except Exception as e:
                print(f"✗ Error processing {video_file}: {str(e)}")

    print("\nProcessing complete!")
