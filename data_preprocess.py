#!/usr/bin/env python
#-*-coding:utf-8 -*-

import numpy as np
import re
from tensorflow.contrib import learn

def clear_str(string):
    #使用正则表达式对原数据进行简单的预处理和分词操作
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_then_label():
    #load movie review pos and neg data, then label them
    positive_data = list(open('./data/rt-polarity.pos', 'r').readlines())
    negative_data = list(open('./data/rt-polarity.neg', 'r').readlines())
    '''
    python中有三个“读”方法，read()\readline()\readlines(), read每次读取整个文件，它通常将文件内容放到一个字符串变量中
    对于连续面向行的处理的情况，这种方式就没必要用了，并且如果文件大于可用内存，则不可能实现这种情况。readline每次只读取一行，
    通常比readlines慢很多，读取时内存占用小，比较适合大文件。readlines自动将文件内容分析成一个行的列表，每行作为一个元素，但
    读取大文件会比较占内存。
    '''
    positive_data = [s.strip() for s in positive_data]
    negative_data = [s.strip() for s in negative_data]
    #s.strip(rm), s为字符串，rm为要删除的字符序列,当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')

    #预处理及分词操作
    positive_data = [clear_str(sentence) for sentence in positive_data]
    negative_data = [clear_str(sentence) for sentence in negative_data]

    #将样本标签化
    label_positive = [[0, 1] for sent in positive_data]
    label_negative = [[1, 0] for sent in negative_data]

    #函数返回预处理后的数据及标签
    data_preprocess = positive_data + negative_data
    data_label = np.concatenate([label_positive, label_negative], 0)
    #np.concatenate能够一次完成多个数组的拼接。

    return [data_preprocess, data_label]

def build_vocabulary():
    data, label = load_data_then_label()
    max_document_length = max([len(x.split(" ")) for x in data])
    vocabulary_proc = learn.preprocessing.VocabularyProcessor(max_document_length)
    digit_data = np.array(list(vocabulary_proc.fit_transform(data)))
    '''
    tf.contrib.learn.preprocessing.VocabularyProcessor
        (max_document_length, min_frequency=0, vocabulary=None, tokenizer_fn=None)
    max_document_length: 文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充。
    min_frequency: 词频的最小值，出现次数小于最小词频则不会被收录到词表中。
    vocabulary: CategoricalVocabulary 对象。
    tokenizer_fn：分词函数
    '''
    return vocabulary_proc, digit_data

def split_train_dev():
    vocabulary_proc, digit_data = build_vocabulary()
    data, label = load_data_then_label()
    # Randomly shuffle data
    np.random.seed(10)
    # random.seed(123456789) # 种子不同，产生的随机数序列也不同，随机数种子都是全局种子
    # 要每次产生随机数相同就要设置种子，相同种子数的Random对象，相同次数生成的随机数字是完全相同的；
    shuffle_indices = np.random.permutation(np.arange(len(label)))
    digit_data_shuffled = digit_data[shuffle_indices]
    label_shuffled = label[shuffle_indices]

    dev_sample_index = -1 * int(0.1 * float(len(label)))    #this is 0.1,of course you can choose other
    data_train, data_dev = digit_data_shuffled[:dev_sample_index], digit_data_shuffled[dev_sample_index:]
    label_train, label_dev = label_shuffled[:dev_sample_index], label_shuffled[dev_sample_index:]
    return data_train, data_dev, label_train, label_dev

def batch_epoch(data, batch_size, num_epochs, shuffle = True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        '''
        在机器学习中，如果训练数据之间相关性很大（例如连续的语音分帧），可能会让结果很差（泛化能力得不到训练）。
        这时通常需要将训练数据打散.在python中可以使用numpy.random.shuffle(x)的方式打散数据，也就是说
        将x中得数据按照随机顺序重排。但是这会遇到一个问题：训练集中包括输入x和输出y。x和y在打散前是一一对应的，
        打散后不能破坏这种对应关系，所以我们可以使用numpy.random.permutation()来帮忙。
        numpy.random.permutation(length)用来产生一个随机序列作为索引，再使用这个序列从原来的数据集中
        按照新的随机顺序产生随机数据集。
        '''
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx : end_idx]
        #yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器。迭代一次遇到yield时就返回yield后面的值。
        # 重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码开始执行。








