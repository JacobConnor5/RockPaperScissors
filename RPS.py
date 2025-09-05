import pandas as pd
import os
import sklearn,pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pygame
import pygame.camera
import time,random

Categories=['Rock','Paper','Scissors']
name = input('what is your name')
score = [[name,0],['CPU',0]]

def img_processor(datadir):
	flat_data_arr=[] #input array will have colour of each pixelin the array
	target_arr=[] #output array will have the answer so thsese colours mean rock

	#print(f'loading... category : {i}')
	path=os.path.join(datadir)
	
	img_array=imread(datadir)
	img_resized=resize(img_array,(150,150,3))
	flat_data_arr.append(img_resized.flatten())

	return flat_data_arr
		
	#print(f'loaded category:{ successfully')

def train():
	
	flat_data_arr=[] #input array will have colour of each pixelin the array
	target_arr=[] #output array will have the answer so thsese colours mean rock
	datadir=''
	#path which contains all the categories of images

	for i in Categories:
		
		print(f'loading... category : {i}')
		path=os.path.join(datadir,i)
		
		for img in os.listdir(path):
			img_array=imread(os.path.join(path,img))
			img_resized=resize(img_array,(150,150,3))
			flat_data_arr.append(img_resized.flatten())
			target_arr.append(Categories.index(i))
		print(f'loaded category:{i} successfully')

	feat=np.array(flat_data_arr)

	print(flat_data_arr)
	print(feat)

	label=np.array(target_arr)

	feat_train,feat_test,label_train,label_test = sklearn.model_selection.train_test_split(feat,label,test_size = 0.1)

	model = KNeighborsClassifier(n_neighbors=3)
	model.fit(feat_train,label_train)
	acc = model.score(feat_test,label_test)
	predictions = model.predict(feat_test)

	for i in range(len(predictions)):
		print('prediction:',predictions[i],'actual:',label_test[i])

	print(acc)

	with open('Rock.pickle','wb') as pickle_file:
		pickle.dump(model,pickle_file)
		pickle_file.close()
	
	return model

def saved_model():
	temp = open('Rock.pickle','rb')
	model = pickle.load(temp)
	
	return model

def camera(model):
	count = ['','ROCK','PAPER','SCISSORS','GO']
	datadir = 'image.jpg'
	pygame.init()
	pygame.camera.init()
	cameras = pygame.camera.list_cameras()

	webcam = pygame.camera.Camera(cameras[0])
	
	webcam.start()
	img = webcam.get_image()

	w = img.get_width()
	h = img.get_height()

	win = pygame.display.set_mode([w,h])
	#surface = pygame.Surface((100,20))

	font = pygame.font.SysFont('Arial',13)
	big_font = pygame.font.SysFont('Roclette',100)

	
	countdown = big_font.render('',True,(0,0,0))
	run = True
	start = time.time()
	current = time.time()

	x = 0

	while run:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

		current = time.time()
		x = round(current-start)
		if x < 0 or x>4:
			x = 0

		countdown = big_font.render(count[x],True,(0,0,0))

		img = webcam.get_image()

		win.blit(img,(0,0))
	
		pygame.image.save(img,datadir)
					
		img = img_processor(datadir)
		prediction = int(model.predict(img))


		text = font.render(Categories[int(prediction)],True,(255,255,255))
		win.blit(text,(0,0))


		win.blit(countdown,(w/2,h/2))
		pygame.display.flip()
		
		#if x == 4:

			
			#start = play(prediction)


def play(prediction):
	choice = random.randint(0,2)
	time.sleep(1)
	print('You chose',Categories[prediction])
	time.sleep(1)
	print('I chose',Categories[choice])

	if choice == prediction+1 or (choice==0 and prediction==2):
		print('You lose')
		score[1][1] += 1

	elif prediction == choice+1 or (prediction == 0 and choice == 2):
		print('Well done',name, 'you won')
		score[0][1] += 1

	start_time =time.time()

	return start_time
	
#train()

model = train()
camera(model)

#recognises rock on the right side of the screen too much (left side of the python window but right side of the camera)
#better on paper and rock than scissors
#really bad a scissors rn
