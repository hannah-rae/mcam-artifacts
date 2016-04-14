import csv
import random
import math
from scipy import misc
import numpy as np
import tensorflow as tf

# Query for retrieving positive label data
# select sol, name, udr_image_id, width, height from udr where udr_image_id in
# 	(select udr_image_id from udr
#        where type_of_product = 'RecoveredProduct'
#        and (instrument = 'ML' or instrument = 'MR')
#        and udr.name not like '%Partial%'
#        and udr.udr_image_id is NOT NULL
#        order by udr_image_id)
# and type_of_product != 'Thumbnail'
# and filter_used = 0
# and sol != 1000
# and width = 168 and height = 150
# and name not like '%Partial%'
# order by udr_image_id

# Query for retrieving negative label data
# select sol, name, udr_image_id, width, height
# from udr 
# where sol > 17 
# 	and type_of_product != 'RecoveredProduct' 
# 	and type_of_product != 'RecoveredThumbnail'
# 	and type_of_product != 'Thumbnail'
# 	and type_of_product != 'Video'
# 	and type_of_product != 'Zstack'
# 	and type_of_product != 'Rangemap'
# 	and sol != 1000
# 	and width = 168 and height = 150
# 	and name not like '%Partial%' 
# 	and filter_used = 0
# 	and (instrument = 'ML' or instrument = 'MR')

def get_array(filename):
	img = misc.imread(filename)
	return img

def get_next_batch(batch_size, batch_num, phase):
	img_list = []
	label_list = []
	offset = batch_size * batch_num
	for i in range(batch_size):
		if phase == 'train':
			img = get_array(train_list[offset + i][0]).flatten()
			label = train_list[offset + i][1]
		else:
			img = get_array(test_list[offset + i][0]).flatten()
			label = test_list[offset + i][1]
		img_list.append(img)
		label_list.append(label)
	return img_list, label_list

def build_url(sol, name):
	if "McamL" in name:
		instrument = "ML"
	elif "McamR" in name:
		instrument = "MR"

	url = "/molokini_raid/MSL/data/surface/processed/images/web/full/SURFACE/" \
			+ instrument + "/" \
			+ "sol" + sol.zfill(4) + "/" \
			+ name.strip("\"") \
			+ ".png"

	return url

# Generate list of images for training set (n positive data)

video_udrs = []
product_dict = dict()
neg_product_dict = dict()
y_pos = [1,0]
y_neg = [0,1]

with open('video-udrs.csv') as f:
	for row in f:
		video_udrs.append(row.rstrip('\n'))

## Filter annoying things out of the list
with open('artifact-image-list.csv') as csvfile:
	for row in csvfile:
		prod = row.split(',')
		sol = prod[0]
		name = prod[1]
		udr_image_id = prod[2].rstrip('\n')
		if udr_image_id in video_udrs:
			continue
		if "RecoveredProduct" in name:
			if udr_image_id in product_dict.keys():
				product_dict[udr_image_id]['recovered-product'] = name
				product_dict[udr_image_id]['rp_sol'] = sol
			else:
				product_dict[udr_image_id] = {'recovered-product': name, \
											  'rp_sol': sol}
		if "Image" in name:
			if udr_image_id in product_dict.keys():
				product_dict[udr_image_id]['image'] = name
				product_dict[udr_image_id]['sol'] = sol
			else:
				product_dict[udr_image_id] = {'image': name, \
											  'sol': sol}


# Build list of pairs: [positive labeled image URL, label]
train_img_list = []
for udr in product_dict.keys():
	if 'image' not in product_dict[udr].keys():
		continue # If there is no image associated with it (for w/e reason), keep going
	url = build_url(product_dict[udr]['sol'], product_dict[udr]['image'])
	train_img_list.append([url, y_pos])

## Generate list of images for training set (n negative data)

# Put crap from csv into dictionary for parsing
with open('nonRP-set.csv') as file:
	for row in file:
		neg_prod = row.split(',')
		neg_udr_image_id = neg_prod[2].rstrip('\n')
		if neg_udr_image_id in neg_product_dict.keys():
			neg_product_dict[neg_udr_image_id]['image'] = neg_prod[1]
			neg_product_dict[neg_udr_image_id]['sol'] = neg_prod[0]
		else:
			neg_product_dict[neg_udr_image_id] = {'image': neg_prod[1], \
												  'sol': neg_prod[0]}


# Build list of pairs: [negative labeled image URL, label]
num_neg_data = int(len(train_img_list)*0.25) # How many negative examples?
used_neg_udrs = []
for i in range(num_neg_data): 
	# Generate random udr until we get one that's not in the positive label list 
	# or already been used for the negative set
	while (True):
		random_udr = neg_product_dict.keys()[random.randint(1, len(neg_product_dict.keys())-1)]
		existing_udrs = product_dict.keys() + used_neg_udrs
		if random_udr in existing_udrs:
			continue
		else:
			used_neg_udrs.append(random_udr)
			break

	# Now we have random_udr not in positive or negative set already
	neg_url = build_url(neg_product_dict[random_udr]['sol'], 
						neg_product_dict[random_udr]['image'])

	train_img_list.append([neg_url, y_neg])

# Shuffle a few times... for good measure
for i in range(5):
	random.shuffle(train_img_list)

# Set aside 30% of the data for test
num_test_data = int(math.floor(len(train_img_list) * .1))
test_list = train_img_list[:num_test_data] # first num_test_data points
train_list = train_img_list[num_test_data:] # num_test_data+1 to N points

print test_list
print train_list
print len(test_list)
print len(train_list)


##########################################################################################
# Now let's make a FF network.
sess = tf.InteractiveSession() # Create a session to run the model in

num_classes = 2
input_vector_size = (150*8)*(168*8)*3

# set up vectors for input x and model parameters W, b
x = tf.placeholder(tf.float32, [None, input_vector_size])
W = tf.Variable(tf.zeros([input_vector_size, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

y_predict = tf.nn.softmax(tf.matmul(x, W) + b) # predicted probability distribution / label
y_true = tf.placeholder(tf.float32, [None, num_classes]) # true probability distribution / label

cross_entropy = -tf.reduce_sum(y_true * tf.log(y_predict)) # set up the cross-entropy cost fn

learning_rate = 1e-4
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

########## TRAINING #############
init = tf.initialize_all_variables() # operation to initialize Variables

sess.run(init) # Run the operation to init Variables in the session

# Stochastic gradient descent to train the model with the cross-entropy loss function
batch_size = 100
# num_batches = 10
num_batches = int(math.floor(len(train_list) / batch_size))

for i in range(num_batches):
	batch_xs, batch_ys = get_next_batch(batch_size, i, 'train')
	if i % 5 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys})
		print("step %d out of %d, training accuracy %g" % (i, num_batches, train_accuracy))
	sess.run(train_step, feed_dict={x:batch_xs, y_true: batch_ys})

print "completed training, beginning testing"

########## TESTING #############

test_batch_size = 200
# num_test_batches = 10
num_test_batches = int(math.floor(len(test_list) / test_batch_size))

for i in range(num_test_batches):
	print "testing batch " + str(i) + " out of " + str(num_test_batches)
	testbatch_xs, testbatch_ys = get_next_batch(test_batch_size, i, 'test')
	print(sess.run(accuracy, feed_dict={x: testbatch_xs, y_true: testbatch_ys}))


















