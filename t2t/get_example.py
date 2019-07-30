import tensorflow as tf

filenames = "~/t2t_data/translation_query-dev-00000-of-00001"
with tf.Graph().as_default():
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

def _load_records(filename):
  return tf.data.TFRecordDataset(
      filename,
      compression_type=tf.constant("GZIP") if False else None,
      buffer_size=16 * 1000 * 1000)

dataset = dataset.flat_map(_load_records)

def _parse_example(ex_ser):
  return tf.parse_single_example(ex_ser, False)

if False:
  dataset = dataset.map(_parse_example, num_parallel_calls=32)
dataset = dataset.prefetch(100)
record_it = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
  while True:
    try:
      ex = sess.run(record_it)
      print(ex)
    except tf.errors.OutOfRangeError:
      break