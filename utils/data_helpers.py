from collections import Counter
import torch
from torch.utils.data import DataLoader
import logging
from .tools import contains_chinese
from .tools import process_cache


class Vocab(object):
    """
    构建词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[num])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['[UNK]'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    :param top_k:  取出现频率最高的前top_k个token
    :param data: 为一个列表，每个元素为一句文本
    :return:
    """
    UNK = '<UNK>'  # 0
    PAD = '<PAD>'  # 1
    BOS = '<BOS>'  # 2
    EOS = '<EOS>'  # 3

    def __init__(self, top_k=2000, data=None, cut_words=False):
        logging.info(f" ## 正在根据训练集构建词表……")
        counter = Counter()
        self.stoi = {Vocab.UNK: 0, Vocab.PAD: 1, Vocab.BOS: 2, Vocab.EOS: 3}
        self.itos = [Vocab.UNK, Vocab.PAD, Vocab.BOS, Vocab.EOS]
        for text in data:
            token = tokenize(text, cut_words)
            counter.update(token)  # 统计每个token出现的频率

        top_k_words = counter.most_common(top_k - len(self.itos))
        for i, word in enumerate(top_k_words):
            self.stoi[word[0]] = i + 4  # 4表示已有了UNK、PAD、BOS和EOS
            self.itos.append(word[0])
        logging.info(f" ## 词表构建完毕，前100个词为: {list(self.stoi.items())[:100]}")

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def tokenize(text, cut_words=False):
    """
    tokenize方法
    :param text: 上联：一夜春风去，怎么对下联？
    :return:
    words: 字粒度： ['上', '联', '：', '一', '夜', '春', '风', '去', '，', '怎', '么', '对', '下', '联', '？']
    """
    import jieba
    if contains_chinese(text):  # 中文
        if cut_words:  # 分词
            text = jieba.cut(text)  # 词粒度
        text = " ".join(text)  # 不分词则是字粒度
    words = text.split()
    return words


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            padding_content = [padding_value] * (max_len - tensor.size(0))
            tensor = torch.cat([tensor, torch.tensor(padding_content)], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class LoadCoupletDataset():
    def __init__(self, train_file_paths=None, batch_size=2, top_k=5000):
        # 根据训练预料建立字典，由于都是中文，所以共用一个即可
        raw_data = self.load_raw_data(train_file_paths)
        self.vocab = Vocab(top_k, raw_data[0] + raw_data[1], False)
        self.PAD_IDX = self.vocab[self.vocab.PAD]
        self.BOS_IDX = self.vocab[self.vocab.BOS]
        self.EOS_IDX = self.vocab[self.vocab.EOS]
        self.batch_size = batch_size
        self.top_k = top_k

    def load_raw_data(self, file_path=None):
        """
        载入原始的文本
        :param file_path:
        :return:
        samples: ['上联：一夜春风去，怎么对下联？', '隋唐五代｜欧阳询《温彦博碑》，书于贞观十一年，是欧最晚的作品']
        labels: ['1','1']
        """
        results = []
        for i in range(2):
            logging.info(f" ## 载入原始文本 {file_path[i]}")
            tmp = []
            with open(file_path[i], encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    tmp.append(line)
            results.append(tmp)
        return results

    @process_cache(["top_k"])
    def data_process(self, filepaths, file_path=None):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        :param filepaths:
        :return:
        """
        results = self.load_raw_data(filepaths)
        data = []
        for (raw_in, raw_out) in zip(results[0], results[1]):
            logging.debug(f"原始上联: {raw_in}")
            in_tensor_ = torch.tensor([self.vocab[token] for token in
                                       tokenize(raw_in.rstrip("\n"))], dtype=torch.long)
            if len(in_tensor_) < 6:
                logging.debug(f"长度过短，忽略: {raw_in}<=>{raw_out}")
                continue
            logging.debug(f"原始上联 token id: {in_tensor_}")
            logging.debug(f"原始下联: {raw_out}")
            out_tensor_ = torch.tensor([self.vocab[token] for token in
                                        tokenize(raw_out.rstrip("\n"))], dtype=torch.long)
            logging.debug(f"原始下联 token id: {out_tensor_}")

            data.append((in_tensor_, out_tensor_))
        return data

    def load_train_val_test_data(self, train_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths, file_path=train_file_paths[0])
        test_data = self.data_process(test_file_paths, file_path=test_file_paths[0])
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, test_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        in_batch, out_batch = [], []
        for (in_item, out_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            in_batch.append(in_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 token
            out = torch.cat([torch.tensor([self.BOS_IDX]), out_item, torch.tensor([self.EOS_IDX])], dim=0)
            out_batch.append(out)
        # 以最长的序列为标准进行填充
        in_batch = pad_sequence(in_batch, padding_value=self.PAD_IDX)  # [de_len,batch_size]
        out_batch = pad_sequence(out_batch, padding_value=self.PAD_IDX)  # [en_len,batch_size]
        return in_batch, out_batch

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, device='cpu'):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)  # [tgt_len,tgt_len]
        # Decoder的注意力Mask输入，用于掩盖当前position之后的position，所以这里是一个对称矩阵

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device)
        # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的，所以这里全是0

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        # 用于mask掉Encoder的Token序列中的padding部分,[batch_size, src_len]
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        # 用于mask掉Decoder的Token序列中的padding部分,batch_size, tgt_len
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def make_inference_sample(self, text):
        tokens = [self.vocab.stoi[token] for token in tokenize(text)]
        num_tokens = len(tokens)
        src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
        return src
