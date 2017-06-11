from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import *
from keras.models import *
from keras.initializers import RandomNormal
import keras.backend as K
import tensorflow as tf



# batch_size=1, InstNorm = BatchNorm?
def InstanceNorm(x):
	mean, var = tf.nn.moments(x, axes=[1,2])
	epsilon = 1e-5
	return (x - mean) * tf.rsqrt(var + epsilon)

def ReflectPad(x, ks=1):
	return tf.pad(x, [[0,0],[ks,ks],[ks,ks],[0,0]], "REFLECT")

def ResBlock(x, dim):
	y = Lambda(ReflectPad)(x)
	y = Conv2D(dim, 3, padding='valid')(y)
	y = LeakyReLU()(y)
	y = Lambda(ReflectPad)(y)
	y = Conv2D(dim, 3, padding='valid')(y)
	y = LeakyReLU()(y)
	return add([x,y])

def ResizeConv(x, cdim):
	def Resize2x(x):
		shape = x.get_shape()
		new_shape = tf.shape(x)[1:3] * tf.constant(np.array([2, 2], dtype='int32'))
		x = tf.image.resize_bilinear(x, new_shape)
		x.set_shape( (None, shape[1]*2, shape[2]*2, None) )
		return x
	x = Lambda(Resize2x)(x)
	x = Lambda(lambda x:ReflectPad(x,1))(x)
	x = Conv2D(cdim, 3, padding='valid')(x)
	return x

def BuildGenerator():
	gdim = 32
	image = Input(shape=(256,256,3))
	x = Lambda(lambda x:ReflectPad(x,3))(image)
	x = Conv2D(gdim,   7, strides=1, padding='valid')(x)
	#x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	x = Lambda(lambda x:ReflectPad(x,1))(x)
	x = Conv2D(gdim*2, 3, strides=2, padding='valid')(x)
	#x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	x = Lambda(lambda x:ReflectPad(x,1))(x)
	x = Conv2D(gdim*4, 3, strides=2, padding='valid')(x)
	#x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	for ii in range(9): x = ResBlock(x, gdim*4)
	x = ResizeConv(x, gdim*2)
	#x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	x = ResizeConv(x, gdim)
	#x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	x = Lambda(lambda x:ReflectPad(x,1))(x)
	x = Conv2D(3, 3, strides=1, padding='valid', activation='tanh')(x)
	return Model(inputs=image, outputs=x)

def BuildDiscriminator():
	cnn = Sequential()
	ddim = 64
	cnn.add(Conv2D(ddim, 7, padding='same', strides=2, input_shape=(256,256,3)))
	cnn.add(BatchNormalization())
	cnn.add(LeakyReLU())
	cnn.add(Conv2D(ddim*2, 3, padding='same', strides=2))
	cnn.add(BatchNormalization())
	cnn.add(LeakyReLU())
	cnn.add(Conv2D(ddim*4, 3, padding='same', strides=2))
	cnn.add(BatchNormalization())
	cnn.add(LeakyReLU())
	cnn.add(Conv2D(ddim*8, 3, padding='same', strides=1))
	cnn.add(BatchNormalization())
	cnn.add(LeakyReLU()) 
	cnn.add(Conv2D(1, 3, padding='same', strides=1))
	image = Input(shape=(256,256,3))
	return Model(inputs=image, outputs=cnn(image))
