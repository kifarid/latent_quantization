import ipdb
import pathlib
import jax.numpy as jnp
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_data(data):
    """
    FeaturesDict({
    'file_name': Text(shape=(), dtype=string),
    'image': Image(shape=(None, None, 3), dtype=uint8),
    'objects': Sequence({
        '3d_coords': Tensor(shape=(3,), dtype=float32),
        'color': ClassLabel(shape=(), dtype=int64, num_classes=8),
        'material': ClassLabel(shape=(), dtype=int64, num_classes=2),
        'pixel_coords': Tensor(shape=(3,), dtype=float32),
        'rotation': float32,
        'shape': ClassLabel(shape=(), dtype=int64, num_classes=3),
        'size': ClassLabel(shape=(), dtype=int64, num_classes=2),
    }),
    'question_answer': Sequence({
        'answer': Text(shape=(), dtype=string),
        'question': Text(shape=(), dtype=string),
    }),
    })"""
    ret = {}
    x = data['image']
    x = x = tf.image.resize(x, [64, 64], antialias=True) #, method='bicubic', 
    #x = tf.reshape(x, (1, 64, 64))
    x = tf.transpose(x, [2, 0, 1])   # (3, 218, 178)
    x = x / 255.
    x = tf.clip_by_value(x, 0., 1.)
    x = tf.cast(x, tf.float32)
    ret['x'] = x
    obj = data['objects']
    per_object = []
    #print(len(objects_seq))
    #for obj in objects_seq:
    #    print(obj)
    obj_feats = tf.concat([
        tf.expand_dims(obj['color'], -1),
        tf.expand_dims(obj['material'], -1),
        tf.cast(obj['pixel_coords'], tf.int64),
        tf.expand_dims(obj['shape'], -1),
        tf.expand_dims(obj['size'], -1)
    ], axis=-1)
    #per_object.append(obj_feats)
    cont_obj_feats = tf.concat([obj['3d_coords'], tf.expand_dims(obj['rotation'], -1)], axis=-1)
    ret['z'] = obj_feats

    ret['cont_z'] = cont_obj_feats
    #ret['num_objects'] = len(o)

    return ret


def get_datasets(config):
    possible_dirs = config.data.possible_dirs
    while len(possible_dirs) > 0:
        possible_dir = pathlib.Path(possible_dirs.pop(0))
        try:
            builder = tfds.builder('clevr', data_dir=possible_dir)
            builder.download_and_prepare()
            break
        except PermissionError as e:
            print(e)
    metadata = {
        'num_train': builder.info.splits['train'].num_examples,
    }

    train_set = builder.as_dataset(split=tfds.Split.TRAIN)
    

    def _filter_obj_count(example):
        return tf.equal(tf.shape(example['z'])[0], config.data.num_objects)
    
    train_set = train_set.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).filter(_filter_obj_count).cache() # .shuffle(100000, seed=config.data.seed).repeat().batch(config.data.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    #flatten the cont_z and z to be 1D
    train_set = train_set.map(lambda x: {**x, 'z': tf.reshape(x['z'], [-1]), 'cont_z': tf.reshape(x['cont_z'], [-1])}, num_parallel_calls=tf.data.AUTOTUNE)
    val_set = train_set.take(config.data.num_val_data)
    #get length of train set
    metadata['num_train'] = len(list(train_set.as_numpy_iterator()))
    metadata['num_val'] = len(list(val_set.as_numpy_iterator()))

    train_set = train_set.shuffle(metadata['num_train'], seed=config.data.seed).repeat().batch(config.data.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_set = val_set.batch(config.data.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return metadata, tfds.as_numpy(train_set), tfds.as_numpy(val_set)


if __name__ == '__main__':
    import omegaconf

    config = omegaconf.OmegaConf.create(
        {
            'data': {
                'possible_dirs': [
                    '/work/dlclarge1/faridk-quantization/data',
                ],
                'seed': 42,
                'batch_size': 2,
                'num_val_data': 1000,
                'num_objects': 4,
            },
        }
    )
    meta, train_set, val_set = get_datasets(config)
    for i, sample in enumerate(train_set):
    
        print(sample['x'].max(), sample['x'].min(), sample['x'].shape, sample['z'].shape)
        if i > 10: 
            break
    train_set = iter(train_set)

