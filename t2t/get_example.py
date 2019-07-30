import tensorflow as tf

def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'inputs': tf.FixedLenFeature([], tf.string),
        'targets': tf.FixedLenSequenceFeature([], tf.int64),
    }
    sample = tf.parse_single_example(data_record, features)
    return sample

# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(["translation_query-dev-00000-of-00001"])
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            data_record = sess.run(next_element)
            print(data_record)
    except:
        raise

