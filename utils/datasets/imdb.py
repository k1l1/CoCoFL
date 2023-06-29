import torch
from torch.utils.data import TensorDataset
from prenlp.data import IMDB as _IMDB
from prenlp.tokenizer import SentencePiece
from collections import OrderedDict


class InputExample:
    """A single training/test example for text classification.
    """
    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label


class InputFeatures:
    """A single set of features of data.
    """
    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


class Tokenizer:
    def __init__(self, tokenizer, vocab_file: str,
                 pad_token: str = "[PAD]",
                 unk_token: str = "[UNK]",
                 bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]"):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab = OrderedDict()
        self.ids_to_tokens = OrderedDict()

        # Build vocab and ids_to_tokens
        with open(vocab_file, "r", encoding="utf-8") as reader:
            for i, line in enumerate(reader.readlines()):
                token = line.split()[0]
                self.vocab[token] = i
        for token, id in self.vocab.items():
            self.ids_to_tokens[id] = token

    def tokenize(self, text: str):
        """Tokenize given text.
        """
        return self.tokenizer(text)

    def convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) in an id (integer) using the vocab.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_id_to_token(self, id: int) -> str:
        """Convert an id (integer) in a token (str) using the vocab.
        """
        return self.ids_to_tokens.get(id, self.unk_token)

    def convert_tokens_to_ids(self, tokens):
        """Convert list of tokens in list of ids using the vocab.
        """
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """Convert list of ids in list of tokens using the vocab.
        """
        return [self.convert_id_to_token(id) for id in ids]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size.
        """
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Id of pad_token in the vocab.
        """
        return self.convert_token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """Id of unk_token in the vocab.
        """
        return self.convert_token_to_id(self.unk_token)

    @property
    def bos_token_id(self) -> int:
        """Id of bos_token in the vocab.
        """
        return self.convert_token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        """Id of eos_token in the vocab.
        """
        return self.convert_token_to_id(self.eos_token)


def convert_examples_to_features(examples,
                                 label_dict: dict,
                                 tokenizer,
                                 max_seq_len: int):
    pad_token_id = tokenizer.pad_token_id

    features = []
    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = tokens[:max_seq_len]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        label_id = label_dict.get(example.label)

        feature = InputFeatures(input_ids, label_id)
        features.append(feature)

    return features


class PretrainedTokenizer(Tokenizer):
    def __init__(self, pretrained_model: str, vocab_file: str,
                 pad_token: str = "[PAD]",
                 unk_token: str = "[UNK]",
                 bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]"):
        tokenizer = SentencePiece.load(pretrained_model)

        super(PretrainedTokenizer, self).__init__(tokenizer, vocab_file, pad_token, unk_token, bos_token, eos_token)

    def detokenize(self, tokens):
        return self.tokenizer.detokenize(tokens)


class IMDB(TensorDataset):
    def __init__(self, root, train=True, seq_len=1024):
        print("Preparing dataset, this might take a while!")
        self.train = train

        self.classes = {"0": "neg", "1": "pos"}
        if train:
            dataset = _IMDB(root=root)[0]
        else:
            dataset = _IMDB(root=root)[1]

        examples = []
        for text, label in dataset:
            example = InputExample(text, label)
            examples.append(example)

        labels = sorted(list(set([example.label for example in examples])))
        label_dict = {label: i for i, label in enumerate(labels)}

        tokenizer = PretrainedTokenizer(pretrained_model="data/sentencepiece.model", vocab_file="data/sentencepiece.vocab")

        features = convert_examples_to_features(examples, label_dict, tokenizer, seq_len)

        all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
        all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)

        self.targets = all_label_ids

        super().__init__(all_input_ids, all_label_ids)