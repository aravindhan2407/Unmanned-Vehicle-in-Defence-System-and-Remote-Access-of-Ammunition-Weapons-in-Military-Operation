from imageai.Detection import ObjectDetection
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path ,"model.h5"))
detector.loadModel()
left = list()
height = list()
label = list()
inputimage = input("Enter the input image name : ")
if(os.path.exists(inputimage)):
	pass
else:
	print("Error!!!Image is not found")
	sys.exit(0)
outputimage = input("Enter the output image name to be created : ")
if(os.path.exists(outputimage)):
	print("Error!!!Image alredy present")
	sys.exit(0)
else:
	detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , inputimage), output_image_path=os.path.join(execution_path , outputimage))
	i=0
	for eachObject in detections:
		label.append(eachObject["name"])
		height.append(eachObject["percentage_probability"])
		left.append(i)
		i+=1
		print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
plt.plot(height,color="blue")
plt.xlabel('Detected Objects')
plt.ylabel('Percentage of correctness in detection')
plt.xticks(rotation=90)
plt.xticks(left,label)
plt.title('Object Detection Chart')
plt.savefig('graph'+os.path.splitext(outputimage)[0]+'.jpg',bbox_inches='tight',pad_inches=1)

