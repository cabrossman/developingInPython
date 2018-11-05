#https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
import os
from pickle import dump
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

os.chdir('C:\\Users\\chrisb\\OneDrive - Leesa\\jobs\\developingInPython\\image_captioning')
data_dir = 'C:\\Users\\chrisb\\Desktop\\Flickr\\'
 
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	#model = model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	
	list_of_pics = os.listdir(directory)
	cnt = 1
	total = len(list_of_pics)
	for name in list_of_pics:
		print(str(cnt) + ' of ' + str(total) + ' : ' + str(name))
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		#print('>%s' % name)
		cnt = cnt + 1
	return features
 
# extract features from all images
directory = 'Flickr8k_Dataset/Flicker8k_Dataset'

features = extract_features(os.path.join(data_dir,directory))
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('data/features.pkl', 'wb'))