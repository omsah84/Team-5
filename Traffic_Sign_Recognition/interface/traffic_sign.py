import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
from keras.models import load_model

# Load model
model = load_model('traffic_classifier.h5')

# Class labels
classes = { 
    1:'Speed limit (20km/h)', 2:'Speed limit (30km/h)', 3:'Speed limit (50km/h)',
    4:'Speed limit (60km/h)', 5:'Speed limit (70km/h)', 6:'Speed limit (80km/h)',
    7:'End of speed limit (80km/h)', 8:'Speed limit (100km/h)', 9:'Speed limit (120km/h)',
    10:'No passing', 11:'No passing veh over 3.5 tons', 12:'Right-of-way at intersection',
    13:'Priority road', 14:'Yield', 15:'Stop', 16:'No vehicles',
    17:'Veh > 3.5 tons prohibited', 18:'No entry', 19:'General caution',
    20:'Dangerous curve left', 21:'Dangerous curve right', 22:'Double curve',
    23:'Bumpy road', 24:'Slippery road', 25:'Road narrows on the right',
    26:'Road work', 27:'Traffic signals', 28:'Pedestrians',
    29:'Children crossing', 30:'Bicycles crossing', 31:'Beware of ice/snow',
    32:'Wild animals crossing', 33:'End speed + passing limits', 34:'Turn right ahead',
    35:'Turn left ahead', 36:'Ahead only', 37:'Go straight or right',
    38:'Go straight or left', 39:'Keep right', 40:'Keep left',
    41:'Roundabout mandatory', 42:'End of no passing', 43:'End no passing veh > 3.5 tons'
}

# Initialize GUI
top = tk.Tk()
top.geometry('900x700')
top.title('Traffic Sign Classification')
top.configure(background='#F0F4F8')
top.minsize(900, 700)

# Style configuration
BUTTON_BG = '#2D4263'
BUTTON_ACTIVE = '#1A2A4A'
TEXT_COLOR = '#FFFFFF'
RESULT_BG = '#FFFFFF'

# Create main container
main_frame = tk.Frame(top, bg='#F0F4F8')
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Heading
heading = tk.Label(main_frame, text="Traffic Sign Recognition", 
                  font=('Arial', 24, 'bold'), bg='#F0F4F8', fg='#2D4263')
heading.pack(pady=20)

# Image display frame
img_frame = tk.Frame(main_frame, bg=RESULT_BG, bd=2, relief=tk.SUNKEN)
img_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# Placeholder text
placeholder = tk.Label(img_frame, text="Upload Traffic Sign Image", 
                      font=('Arial', 14), fg='#7D8BA5', bg=RESULT_BG)
placeholder.pack(expand=True, fill=tk.BOTH)

sign_image = tk.Label(img_frame, bg=RESULT_BG)
sign_image.pack_forget()

# Result label
result_frame = tk.Frame(main_frame, bg=RESULT_BG, bd=2, relief=tk.SUNKEN)
result_frame.pack(fill=tk.X, pady=10)
label = tk.Label(result_frame, text='', font=('Arial', 16, 'bold'), 
                bg=RESULT_BG, fg=BUTTON_BG, pady=10)
label.pack()

def classify(file_path):
    try:
        with Image.open(file_path) as img:
            # Preprocess image
            img = img.resize((30, 30), Image.Resampling.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            pred = np.argmax(model.predict(img_array), axis=1)[0]
            sign = classes.get(pred + 1, "Unknown sign")
            label.config(text=sign)
    except Exception as e:
        messagebox.showerror("Error", f"Classification failed: {str(e)}")

def show_classify_button(file_path):
    classify_btn = tk.Button(main_frame, text="Analyze Image", 
                            command=lambda: classify(file_path),
                            bg=BUTTON_BG, fg=TEXT_COLOR, activebackground=BUTTON_ACTIVE,
                            font=('Arial', 12, 'bold'), padx=20, pady=8, cursor="hand2")
    classify_btn.pack(pady=15)
    
    # Hover effects
    classify_btn.bind("<Enter>", lambda e: classify_btn.config(bg=BUTTON_ACTIVE))
    classify_btn.bind("<Leave>", lambda e: classify_btn.config(bg=BUTTON_BG))

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            with Image.open(file_path) as img:
                # Fix image orientation
                img = ImageOps.exif_transpose(img)
                img.thumbnail((400, 400))
                
                # Update display
                imgtk = ImageTk.PhotoImage(img)
                placeholder.pack_forget()
                sign_image.config(image=imgtk)
                sign_image.image = imgtk
                sign_image.pack(expand=True)
                label.config(text='')
                show_classify_button(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

# Upload button
upload_btn = tk.Button(main_frame, text="Upload Image", command=upload_image,
                      bg=BUTTON_BG, fg=TEXT_COLOR, activebackground=BUTTON_ACTIVE,
                      font=('Arial', 12, 'bold'), padx=20, pady=8, cursor="hand2")
upload_btn.pack(pady=15)

# Hover effects
upload_btn.bind("<Enter>", lambda e: upload_btn.config(bg=BUTTON_ACTIVE))
upload_btn.bind("<Leave>", lambda e: upload_btn.config(bg=BUTTON_BG))

top.mainloop()