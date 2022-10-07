import numpy as np
from pathlib import Path
from tensorflow import keras


def is_valid_captcha(captcha):
    data_dir = Path("../input/russian-captcha-images-base64/img/answers")

    characters = set()
    # A list to store the length of each captcha
    captcha_length = []

    # Store image-label info
    dataset = []

    # Get list of all the images
    images = list(data_dir.glob("*.jpg"))

    for img_path in images:
        # 1. Get the label associated with each image
        label = img_path.name.split(".jpg")[0]
        # 2. Store the length of this captcha
        captcha_length.append(len(label))
        # 3. Store the image-label pair info
        dataset.append((str(img_path), label))

        # 4. Store the characters present
        for ch in label:
            characters.add(ch)

    for ch in captcha:
        if not ch in characters:
            return False
    return True


class DataGenerator(keras.utils.Sequence):
    """Generates batches from a given dataset.

    Args:
        data: training or validation data
        labels: corresponding labels
        char_map: dictionary mapping char to labels
        batch_size: size of a single batch
        img_width: width of the resized
        img_height: height of the resized
        downsample_factor: by what factor did the CNN downsample the images
        max_length: maximum length of any captcha
        shuffle: whether to shuffle data or not after each epoch
    Returns:
        batch_inputs: a dictionary containing batch inputs
        batch_labels: a batch of corresponding labels
    """

    def __init__(self,
                 data,
                 labels,
                 char_map,
                 batch_size=16,
                 img_width=200,
                 img_height=50,
                 downsample_factor=4,
                 max_length=5,
                 shuffle=True
                 ):
        self.data = data
        self.labels = labels
        self.char_map = char_map
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.downsample_factor = downsample_factor
        self.max_length = max_length
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        # 1. Get the next batch indices
        curr_batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 2. This isn't necessary, but it can help us save some memory
        # as not all batches the last batch may not have elements
        # equal to the batch_size
        batch_len = len(curr_batch_idx)

        # 3. Instantiate batch arrays
        batch_images = np.ones((batch_len, self.img_width, self.img_height, 1),
                               dtype=np.float32)
        batch_labels = np.ones((batch_len, self.max_length), dtype=np.float32)
        input_length = np.ones((batch_len, 1), dtype=np.int64) * \
                       (self.img_width // self.downsample_factor - 2)
        label_length = np.zeros((batch_len, 1), dtype=np.int64)

        for j, idx in enumerate(curr_batch_idx):
            # 1. Get the image and transpose it
            img = self.data[idx].T
            # 2. Add extra dimension
            img = np.expand_dims(img, axis=-1)
            # 3. Get the corresponding label
            text = self.labels[idx]
            # 4. Include the pair only if the captcha is valid
            if is_valid_captcha(text):
                label = [self.char_map[ch] for ch in text]
                batch_images[j] = img
                batch_labels[j] = label
                label_length[j] = len(text)

        batch_inputs = {
            'input_data': batch_images,
            'input_label': batch_labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        return batch_inputs, np.zeros(batch_len).astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
