import random
import tensorflow as tf

def serialize_example(slots, sign_pool):
    fea_desc = {}

    fea_desc["uniq_id"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"xxxxx"]))
    label = random.choice([1, 1, 0])
    fea_desc["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

    for slot, slot_len in slots:
        # test default value for not exist feature
        if label == 0 and slot == 4:
            continue

        p = sign_pool[slot]
        values = []
        for i in range(slot_len):
            values.append(random.choice(p))

        fea_desc[slot] = tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    example_proto = tf.train.Example(features=tf.train.Features(feature=fea_desc))

    return example_proto.SerializeToString()

def generate_data(name):
    slots = [("1", 1), ("2", 1), ("3", 1), ("4", 3)]
    count = 12000

    sign_pool = {}

    for slot, slot_len in slots:
        sign_pool[slot] = [random.randint(1, 2**63) for i in range(10)]

    # test slot 1 as dense feature
    sign_pool["1"] = [1, 2, 3]

    with tf.io.TFRecordWriter("./data/tf-part.%s" % name) as writer:
        for i in range(count):
            example = serialize_example(slots, sign_pool)
            writer.write(example)

generate_data('00002')
