from util import *
from decoder import subsequent_mask
from Transformer import make_model
import os
from torch.nn.utils.rnn import pad_sequence


class Batch:
    def __init__(self, src, trg=None, pad=0):
        """
        定义一个批处理对象，其中包含用于训练的源句子和目标句子，以及构建掩码。
        :param src: (Tensor)
        :param trg: (Tensor)
        :param pad:
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        生成一个mask来隐藏填充将来出现的词
        :param tgt:
        :param pad:
        :return:
        """
        # 在倒数第二个维度上增加一个维度
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        """
        计算损失的类
        :param generator: 模型的生成器，即transformer最后的预测输出层，对应linea+softmax
        :param criterion: 标签平滑的惩罚项
        :param opt: 优化器
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data.[0] * norm
        return loss.data.item() * norm
######################################################################################################################
# @例子2
# 给定来自小词汇表的一组文字，目标是翻译，称之为translate task
######################################################################################################################
def load_data():
    # 为了翻译任务，添加特殊的开始和结束标记
    src_vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "how": 5, "are": 6, "you": 7}
    tgt_vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "bonjour": 3, "monde": 4, "comment": 5, "ça": 6, "va": 7}

    src_vocab_reverse = {v: k for k, v in src_vocab.items()}
    tgt_vocab_reverse = {v: k for k, v in tgt_vocab.items()}

    english_sentences = ["hello world", "how are you"]
    french_sentences = ["bonjour monde", "comment ça va"]

    src_data = [[src_vocab[w] for w in ("<s> " + sent + " </s>").split()] for sent in english_sentences]
    tgt_data = [[tgt_vocab[w] for w in ("<s> " + sent + " </s>").split()] for sent in french_sentences]

    return src_data, tgt_data, src_vocab, tgt_vocab, src_vocab_reverse, tgt_vocab_reverse


def translate(model, src, src_vocab, tgt_vocab_reverse, src_tokenizer):
    model.eval()
    tokens = [src_vocab[token] for token in src_tokenizer(src)]
    src = torch.LongTensor(tokens).reshape(1, -1)
    src_mask = (src != src_vocab["<pad>"]).unsqueeze(-2)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(src_vocab["<s>"]).type_as(src.data)
    for i in range(len(tokens) + 5):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == tgt_vocab["</s>"]:
            break
    # 安全地从索引获取单词，对缺失的索引使用默认值
    translated_sentence = ' '.join([tgt_vocab_reverse.get(idx.item(), '<unk>') for idx in ys.flatten()[1:] if idx not in [src_vocab["<s>"], src_vocab["</s>"], src_vocab["<pad>"]]])
    return translated_sentence





def data_gen(src_data, tgt_data, batch_size):
    num_batches = len(src_data) // batch_size

    for i in range(num_batches):
        # 切片操作获取每个批次的源数据和目标数据
        src_batch = [torch.tensor(src_data[j], dtype=torch.long) for j in range(i * batch_size, (i + 1) * batch_size)]
        tgt_batch = [torch.tensor(tgt_data[j], dtype=torch.long) for j in range(i * batch_size, (i + 1) * batch_size)]

        # 使用 pad_sequence 来填充序列，batch_first=True 使批次维度为第一个维度
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

        batch = Batch(src_batch, tgt_batch, 0)
        print(f"Batch {i}: src size = {src_batch.size()}, tgt size = {tgt_batch.size()}, ntokens = {batch.ntokens}")
        yield Batch(src_batch, tgt_batch, 0)


if __name__ == "__main__":
    print("Training starts.")
    src_data, tgt_data, src_vocab, tgt_vocab, src_vocab_reverse, tgt_vocab_reverse = load_data()
    model = make_model(len(src_vocab), len(tgt_vocab), N=2)
    criterion = LabelSmoothing(size=len(tgt_vocab), padding_idx=0, smoothing=0.1)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        model.train()
        run_epoch(data_gen(src_data, tgt_data, 2), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        run_epoch(data_gen(src_data, tgt_data, 2), model, SimpleLossCompute(model.generator, criterion, None))
    print("Training completed.")

    src_sentence = "how are you"
    translated_sentence = translate(model, src_sentence, src_vocab, tgt_vocab_reverse, lambda x: x.split())
    print(f"Original: {src_sentence}, Translated: {translated_sentence}")

