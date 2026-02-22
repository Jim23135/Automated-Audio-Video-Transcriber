import os
import shutil
import torch
import whisper
import subprocess

#
# Variables

# Adds an extra blank line after each transcription line
easierReadingOutput = False

#
# Internal variables
relativePath = os.path.dirname(os.path.realpath(__file__))

inputs = os.path.join(relativePath, "inputs")
inputMp3s = os.path.join(inputs, "input_mp3s")
inputMp4s = os.path.join(inputs, "input_mp4s")

tempOutputs = os.path.join(relativePath, "temp_outputs")
tempOutputMp3s = os.path.join(tempOutputs, "temp_mp3s")
tempOutputText = os.path.join(tempOutputs, "temp_transcriptions")

finalOutputs = os.path.join(relativePath, "final_outputs")

def mp4ToMp3(mp4Path, mp3Path):   
    command = [
        "ffmpeg",
        "-i", mp4Path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "0",
        mp3Path
    ]

    try:
        subprocess.run(command, check=True)

        return mp3Path
    except subprocess.CalledProcessError as e:
        print(f"Error converting MP4 to MP3: {e}")

        return None

def convertSecondToHHMMSS(inputSeconds):
    inputSeconds = int(inputSeconds)

    hours = inputSeconds // 3600
    minutes = (inputSeconds % 3600) // 60
    seconds = inputSeconds % 60

    if hours == 0:
        return f"{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

#
# Main code
if __name__ == "__main__":
    # Create each folder if it doesn't exist
    folders = [inputs, inputMp3s, inputMp4s, tempOutputs, tempOutputMp3s, tempOutputText, finalOutputs]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)    

    print("\nConverting MP4 files...")
    mp4FilesConverted = 0
    for mp4File in os.listdir(inputMp4s):
        print(mp4File)
        if mp4File.lower().endswith(".mp4"):
            if mp4ToMp3(os.path.join(inputMp4s, mp4File), os.path.join(tempOutputMp3s, mp4File[:-4] + ".mp3")):
                mp4FilesConverted += 1
    print(f"Converted {mp4FilesConverted} MP4 files to MP3s")

    # Move mp3s to processing folder
    for mp3File in os.listdir(inputMp3s):
        if mp3File.lower().endswith(".mp3"):
            shutil.move(os.path.join(inputMp3s, mp3File), os.path.join(tempOutputMp3s, mp3File))

    # Load model
    print("\nLoading model...")
    if torch.cuda.is_available():
        model = whisper.load_model("small", device="cuda")
        print("Model loaded and is using GPU")
    else:
        model = whisper.load_model("small", device="cpu")
        print("Model loaded and is using CPU")

    print("\nTranscribing MP3 files.")
    for mp3File in os.listdir(tempOutputMp3s):
        if len(mp3File) < 4:
            continue
        
        print(f"Starting transcription on {mp3File}")
        
        result = model.transcribe(os.path.join(tempOutputMp3s, mp3File))

        print("Finished transcription")

        with open(os.path.join(tempOutputText, mp3File[:-4] + ".txt"), "w") as transcriptionOutputFile:
            for segment in result["segments"]:
                transcriptionOutputFile.write(f"{convertSecondToHHMMSS(segment['start'])} - {convertSecondToHHMMSS(segment['end'])}: {segment['text']}\n" + ("\n" if easierReadingOutput else ""))
    
    for folder in [inputMp4s, tempOutputMp3s, tempOutputText]:
        for item in os.listdir(folder):
            if len(item) < 4:
                continue

            fullOutputPathForItem = os.path.join(finalOutputs, item[:-4])

            # Make an output folder for this file in the output folders if it wasn't already there
            if not os.path.exists(fullOutputPathForItem):
                os.makedirs(fullOutputPathForItem)
            
            if os.path.exists(os.path.join(folder, item)):
                shutil.move(os.path.join(folder, item), fullOutputPathForItem)
