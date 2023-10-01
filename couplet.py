from config import Config
from model import CoupletModel
from utils import LoadCoupletDataset
import torch
import numpy as np


def sampling_decode(model, src, max_len, start_symbol, config=None):
    src = src.to(config.device)
    memory = model.encoder(src)  # 对输入的Token序列进行解码翻译
    ys = torch.ones(1, 1).fill_(start_symbol). \
        type(torch.long).to(config.device)  # 解码的第一个输入，起始符号
    for i in range(max_len):
        memory = memory.to(config.device)
        out = model.decoder(ys, memory)  # # [tgt_len,1,embed_dim]
        out = out.transpose(0, 1)  # [1,tgt_len, embed_dim]
        prob = model.classification(out[:, -1])  # 只对对预测的下一个词进行分类
        # out[:,1] shape : [1,embed_dim],  prob shape:  [1,tgt_vocab_size]
        # _, next_word = torch.max(prob, dim=1)  # 方法1： 选择概率最大者
        # next_word = next_word.item()
        # 方法2： 按概率采样
        prob = torch.softmax(prob.reshape(-1), dim=-1).numpy()
        next_word = np.random.choice(len(prob), p=prob)
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # 将当前时刻解码的预测输出结果，同之前所有的结果堆叠作为输入再去预测下一个词。
    return ys.flatten()


def couplet(model, src, data_loader, config):
    vocab = data_loader.vocab
    model.eval()
    src = data_loader.make_inference_sample(src)
    tgt_tokens = sampling_decode(model, src, max_len=src.shape[0],
                               start_symbol=data_loader.BOS_IDX, config=config) # 解码的预测结果
    result = "".join([vocab.itos[tok] for tok in tgt_tokens])
    return result.replace(data_loader.vocab.BOS, "").replace(data_loader.vocab.EOS, "")


def do_couplet(srcs, config):
    data_loader = LoadCoupletDataset(config.train_corpus_file_paths,
                                     batch_size=config.batch_size,
                                     top_k=config.top_k)
    couplet_model = CoupletModel(vocab_size=len(data_loader.vocab),
                                 d_model=config.d_model,
                                 nhead=config.num_head,
                                 num_encoder_layers=config.num_encoder_layers,
                                 num_decoder_layers=config.num_decoder_layers,
                                 dim_feedforward=config.dim_feedforward,
                                 dropout=config.dropout)
    couplet_model = couplet_model.to(config.device)
    loaded_paras = torch.load(config.model_save_path)
    couplet_model.load_state_dict(loaded_paras)
    results = []
    for src in srcs:
        r = couplet(couplet_model, src, data_loader, config)
        results.append(r)
    return results


if __name__ == '__main__':
    srcs = ["晚风摇树树还挺",
            "忽忽几晨昏，离别间之，疾病间之，不及终年同静好",
            "风声、雨声、读书声，声声入耳"]
    tgts = ["晨露润花花更红",
            "茕茕小儿女，孱羸若此，娇憨若此，更烦二老费精神",
            "家事、国事、天下事，事事关心"]
    config = Config()
    srcs = [" ".join(src) for src in srcs]
    results = do_couplet(srcs, config)
    for src, tgt, r in zip(srcs, tgts, results):
        print(f"上联：{''.join(src.split())}")
        print(f" AI：{r}")
        print(f"下联：{tgt}")
        print("=======")