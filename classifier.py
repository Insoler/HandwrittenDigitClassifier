import numpy as np
import time
class NeuralNet(object):
	def __init__(self):
		self.inputlayer=784
		self.hidden1=200
		self.hidden2=200
		self.hidden3=200
		self.outputlayer=10
		self.theta1=2*np.random.random((784,200))-1
		self.theta2=2*np.random.random((200,200))-1
		self.theta3=2*np.random.random((200,200))-1
		self.theta4=2*np.random.random((200,10))-1

	def activate(self,z,deriv=False):
		if deriv==True:
			return z*(1-z)
		else:
			return 1/(1+np.exp(-z))
	def train(self,inputs,outputs):
		for _ in range(1000):
			l2=self.activate(np.dot(inputs,self.theta1))
			l3=self.activate(np.dot(l2,self.theta2))
			l4=self.activate(np.dot(l3,self.theta3))
			l5=self.activate(np.dot(l4,self.theta4))
			l5_error=outputs-l5
			l5_delta=l5_error*self.activate(l5,True)
			l4_error=np.dot(l5_delta,self.theta4.T)
			l4_delta=l4_error*self.activate(l4,True)
			l3_error=np.dot(l4_delta,self.theta3.T)
			l3_delta=l3_error*self.activate(l3,True)
			l2_error=np.dot(l3_delta,self.theta2.T)
			l2_delta=l2_error*self.activate(l2,True)
			self.theta4+=np.dot(l4.T,l5_delta)
			self.theta3+=np.dot(l3.T,l4_delta)
			self.theta2+=np.dot(l2.T,l3_delta)
			self.theta1+=np.dot(inputs.T,l2_delta)
	def predict(self,inputs):
		l2=self.activate(np.dot(inputs,self.theta1))
		l3=self.activate(np.dot(l2,self.theta2))
		l4=self.activate(np.dot(l3,self.theta3))
		l5=self.activate(np.dot(l4,self.theta4))
		print(l5)

if __name__ == '__main__':
	f = open("mnist_train.csv", 'r')
	a = f.readlines()
	f.close()
	count=0
	inputs=[]
	output=[]
	for line in a:
	    linebits = line.split(',')
	    linebits=list(map(int,linebits[1:]))
	    inputs.append(linebits[1:])
	    output.append(linebits[0])
	for x in inputs:
		x.append(1)
	inputs=np.array(inputs)
	outputs=[[0 for i in range(10)] for i in range(10000)]
	for x,y in zip(output,outputs):
		y[int(x)]=1
	outputs=np.array(outputs).reshape(10000,10)
	nn=NeuralNet()
	nn.train(inputs, outputs)
	print("neural net has been trained")
	for i,_ in enumerate(inputs):
		nn.predict([[inputs[i]]])
		print(outputs[i])
		time.sleep(1)