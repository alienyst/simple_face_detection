import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np

#model pretrained
FASE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier('haarcascade_smile.xml')

#fungsi face detect face
def detect_face(Image):
	#convert to rgb
	new_image = np.array(Image.convert('RGB'))
	# menghapus alpha channel
	img = cv2.cvtColor(new_image, 1)
	# mengubah ke gray
	gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

	# mendeteksi object gambar dalam berbeda ukuran input
	faces = FASE_CASCADE.detectMultiScale(gray, 1.1,1)

	# mengembalikan list of rectangle / menemukan wajah
	for (x,y,w,h) in faces :
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

		# region of interest
		roi_gray = gray[y:y+h, x:x+h]
		roi_color = img[y:y+h, x:x+h]

		#looping eye dan smile
		eyes = EYE_CASCADE.detectMultiScale(roi_gray)

		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

		smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 2, 4)

		for (sx, sy, sw, sh) in smiles:
			cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)

	return img, faces


#tampilan
def main():
	"""Face Image Detection APP"""
	st.title("Face Image Detection App")
	st.text("Build with Streamlit and OpenCV")

	activities = ['Home', 'About']
	choice = st.sidebar.selectbox("Select Activity", activities)

	if choice == 'Home':
		st.subheader('Face Detection')

		image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

		if image_file is not None:
			image = Image.open(image_file)
			st.text("Original Image")
			st.image(image)

		task = ["Face Detection"]
		feature_choice = st.sidebar.selectbox("Task", task)
		if st.button('Process'):
			if feature_choice == 'Face Detection':
				result_img, result_face = detect_face(image)
				st.success(f"Found {len(result_face)} faces.")
				st.image(result_img)



	elif choice == 'About':
		st.subheader("About Face Detection App")
		st.markdown('Build with Streamlit and OpenCV as a project')
		st.text('On Progress')

if __name__ == '__main__' :
 	main()