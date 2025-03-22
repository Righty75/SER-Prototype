import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import scipy.ndimage
from keras.models import load_model
import pygame

# Ensure correct file paths when running as an exe
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Load the trained model (ensure it's in the exe directory)
model_path = get_resource_path("speech_rec_model.h5")

if not os.path.exists(model_path):
    messagebox.showerror("Error", "Model file 'speech_rec_model.h5' not found!")
    sys.exit()

model = load_model(model_path)
print("Model loaded successfully!")

class_names = ["โกรธ", "ขับข้องใจ", "มีความสุข", "ปกติ", "เศร้า"]

pygame.mixer.init()

# Extract MFCC features and resize
def extract_mfcc_resized(filename, target_length=130):
    try:
        y, sr = librosa.load(filename, sr=None)  # Load full audio (MP3 & WAV)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCCs
        mfcc_resized = scipy.ndimage.zoom(mfcc, (1, target_length / mfcc.shape[1]), order=2)  # Resize MFCC
        return mfcc_resized  
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process audio file: {e}")
        return None

# Predict emotion, top 3 predictions
def predict_emotion(model, audio_path):
    mfcc_resized = extract_mfcc_resized(audio_path)
    if mfcc_resized is None:
        return None

    try:
        mfcc_resized = np.expand_dims(mfcc_resized, axis=(0, -1))
        predictions = model.predict(mfcc_resized)[0]  
        top3_indices = np.argsort(predictions)[::-1][:3]  
        results = [(class_names[idx], predictions[idx] * 100) for idx in top3_indices]  
        return results
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        return None

# Select an audio file
def select_audio_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])  # WAV & MP3
    if file_path:
        entry_audio_path.delete(0, tk.END)  
        entry_audio_path.insert(0, file_path)  

# Play selected audio file
def play_audio():
    file_path = entry_audio_path.get()
    if not file_path:
        messagebox.showwarning("ไม่มีการเลือกไฟล์เสียง", "โปรดเลือกไฟล์เสียงก่อน")
        return
    
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to play audio: {e}")

# Display predictions
def analyze_audio():
    file_path = entry_audio_path.get()  
    if not file_path:
        messagebox.showwarning("ไม่มีการเลือกไฟล์เสียง", "โปรดเลือกไฟล์เสียงก่อน")
        return

    try:
        results = predict_emotion(model, file_path)  
        if results is None:
            return  

        result_text.config(state=tk.NORMAL)  
        result_text.delete(1.0, tk.END)  

        result_text.insert(tk.END, "อารมณ์ 3 อันดับแรก :\n")
        for i, (emotion, confidence) in enumerate(results):
            result_text.insert(tk.END, f"{i+1}. {emotion} ({confidence:.2f}%)\n")

        result_text.config(state=tk.DISABLED)  
    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyze audio: {e}")

def clear_results():
    entry_audio_path.delete(0, tk.END)  
    result_text.config(state=tk.NORMAL)  
    result_text.delete(1.0, tk.END)  
    result_text.config(state=tk.DISABLED)  

# =======================
#       Tkinter UI
# =======================

root = tk.Tk()
root.title("Speech Emotion Recognition")
root.geometry("900x500")
root.configure(bg="white")

# File selection
frame_top = tk.Frame(root, bg="white")
frame_top.pack(pady=20)

tk.Label(frame_top, text="เลือกไฟล์เสียงที่ต้องการ(MP3, WAV) : ", font=("Kanit", 12), bg="white").grid(row=0, column=0, padx=10)
entry_audio_path = tk.Entry(frame_top, width=60, font=("Kanit", 10))
entry_audio_path.grid(row=0, column=1, padx=10)
tk.Button(frame_top, text="เลือกไฟล์", command=select_audio_file).grid(row=0, column=2, padx=10)

# buttons
frame_buttons = tk.Frame(root, bg="white")
frame_buttons.pack(pady=15)

tk.Button(frame_buttons, text="ทำนายอารมณ์", font=("Kanit"), command=analyze_audio, height=2, width=15).grid(row=0, column=0, padx=20)
tk.Button(frame_buttons, text="ล้าง", font=("Kanit"), command=clear_results, height=2, width=10).grid(row=0, column=1, padx=20)
tk.Button(frame_buttons, text="เล่นเสียง", font=("Kanit"), command=play_audio, height=2, width=10).grid(row=0, column=2, padx=20)  # Play button

# Result display
frame_results = tk.Frame(root, bg="white")
frame_results.pack(pady=20)

tk.Label(frame_results, text="ผลลัพธ์", font=("Kanit", 12), bg="white").pack()
result_text = tk.Text(frame_results, height=8, width=60, font=("Kanit", 12), state=tk.DISABLED)
result_text.pack()

root.mainloop()