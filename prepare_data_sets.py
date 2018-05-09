#taken from https://github.com/zillding/MRS5228

import os

n = 10
input_file_path = './data/movie_reviews.csv'
output_dir = './data/sets/'



input_file = open(input_file_path, 'rb')

# prepare output files
output = []
for i in range(n):
    # create the output dir if it does not exist
    dirname = output_dir + `i`
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    dict = {
        'train': open(dirname + '/train.csv', 'wb'),
        'test': open(dirname + '/test.csv', 'wb')
    };
    output.append(dict)

# process data and write to output
count = 0
for line in input_file:
    count += 1

    if count == 1:
        # write header to every file
        for dict in output:
            dict['train'].write(line)
            dict['test'].write(line)
        continue

    # determine where should this line go
    test_index = count % n
    for index, dict in enumerate(output):
        if index == test_index:
            dict['test'].write(line)
        else:
            dict['train'].write(line)


# close files
input_file.close()
for dict in output:
    dict['train'].close()
    dict['test'].close()
