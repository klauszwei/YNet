import os
import glob
import argparse
import torch
import librosa
import soundfile as sf
from rich.progress import track

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./g_best_dns.pt")
    parser.add_argument("--input_noisy_wavs_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path, map_location=device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    wav_list = glob.glob(os.path.join(args.input_noisy_wavs_dir, "*.wav"))

    with torch.no_grad():
        for wav_path in track(wav_list):
            wav, _ = librosa.load(wav_path, sr=args.sr)
            wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)

            enhanced = model(wav_tensor)

            out_path = os.path.join(args.output_dir, os.path.basename(wav_path))
            sf.write(out_path, enhanced.squeeze().cpu().numpy(), args.sr, "PCM_16")

    print("Done. Enhanced wavs saved to:", args.output_dir)


if __name__ == "__main__":
    main()