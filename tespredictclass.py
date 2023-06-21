from PIL import Image
import numpy as np

image_path = 'C:\Users\baruz\OneDrive\Documents\AB B\CobaPCA\img\Botol1.jpg'
image = Image.open(image_path).resize((100, 100))  # Resize the image
image = np.array(image)  # Convert image to a numpy array
image = image.astype('float32') / 255.0  # Normalize the pixel values
image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size

predictions = model.predict(image)
predicted_class_index = np.argmax(predictions)
class_labels = ['Botol', 'Masker', 'Kertas', 'Kardus', 'Plastik', 'Kaleng', 'Gelas']  # Replace with your class labels
predicted_class_label = class_labels[predicted_class_index]

print(f'Predicted class: {predicted_class_label}')