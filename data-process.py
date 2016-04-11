import csv
import random
import math

def build_url(sol, name):
	# TODO: Fix bug where neither case is true...
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
				product_dict[udr_image_id]['sol'] = sol
			else:
				product_dict[udr_image_id] = {'recovered-product': name, 'sol': sol}
		if "Image" in name:
			if udr_image_id in product_dict.keys():
				product_dict[udr_image_id]['image'] = name
				product_dict[udr_image_id]['sol'] = sol
			else:
				product_dict[udr_image_id] = {'image': name, 'sol': sol}

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
			neg_product_dict[neg_udr_image_id] = {'image': neg_prod[1], 'sol': neg_prod[0]}

# Build list of pairs: [negative labeled image URL, label]
num_pos_data = len(train_img_list)
used_neg_udrs = []
for i in range(num_pos_data): # Want as many negative examples as positive
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
num_test_data = int(math.floor(len(train_img_list) * .3))
test_list = train_img_list[:num_test_data]
train_list = train_img_list[num_test_data:]

print test_list
print train_list

