import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import numpy as np

canvas_width, canvas_height = 280, 280  # 10x scale

class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, bg='white')
        self.canvas.pack()
        self.button = tk.Button(self, text="Predict", command=self.predict)
        self.button.pack()
        self.clear_btn = tk.Button(self, text="Clear", command=self.clear)
        self.clear_btn.pack()

        self.image = Image.new("L", (canvas_width, canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.model = model

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)

    # def predict(self):
    #     img_resized = self.image.resize((28, 28))
    #     img_array = np.array(img_resized)
    #     img_array = img_array / 255.0
    #     img_array = img_array.reshape(1, 28, 28)
    #     prediction = self.model.predict(img_array)
    #     digit = np.argmax(prediction)
    #     messagebox.showinfo("Prediction", f"Predicted Digit: {digit}")
    def predict(self):
        img_resized = self.image.resize((28, 28))
        img_array = np.array(img_resized)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        messagebox.showinfo("Prediction", f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%")

# Load the model
model = tf.keras.models.load_model("digit_model.h5")

# Run the app
app = App(model)
app.mainloop()
