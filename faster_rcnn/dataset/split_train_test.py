'''
划分 VOC 数据集:
将 VOC 数据集， 划分成 train.txt, test.txt, 其记录 train, test 的 file_name
'''

import os
import random


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "../data/annotations"

    val_rate = 0.3

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    train_f = open("../data/train.txt", "x")
    eval_f = open("../data/val.txt", "x")
    train_f.write("\n".join(train_files))
    eval_f.write("\n".join(val_files))


if __name__ == '__main__':
    main()