import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, func, tokenizer, max_length, texts_1, texts_2, labels):
        self.tokenizer = tokenizer
        self.func = func
        self.max_length = max_length
        self.texts_1 = texts_1
        self.texts_2 = texts_2
        self.labels = labels

    def _encode(self, text):
        return self.func(text, self.tokenizer, self.max_length)

    def __getitem__(self, idx):
        text_1 = self.texts_1[idx]
        text_2 = self.texts_2[idx]
        label = self.labels[idx]
        encoding_1 = self._encode(text_1)
        encoding_2 = self._encode(text_2)

        item = {}
        for key, val in encoding_1.items():
            item[key + '_1'] = torch.tensor(val)
        for key, val in encoding_2.items():
            item[key + '_2'] = torch.tensor(val)
        item['labels'] = torch.tensor(label)
        return item

    def __len__(self):
        return len(self.labels)

