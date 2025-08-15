import random


def _split(_sample_path, _train_path, _test_path):
    """
    读取数据集，将0/1的数据打散，随机选择，按照3：1的比例，选择训练集、测试集
    """
    fp_train = open(_train_path, 'w', encoding='utf-8')
    fp_test  = open(_test_path, 'w', encoding='utf-8')
    with open(_sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                _text_label_pair = line.strip().split("\t")
                if len(_text_label_pair) == 2:
                    rd = random.random()
                    if rd > 0.25:
                        fp_train.write(line.strip() + "\n")
                    else:
                        fp_test.write(line.strip() + "\n")
    fp_train.close()
    fp_test.close()
    f.close()


if __name__ == "__main__":
    sample_path = "../data/intention_data.txt"
    train_path = "../data/intention_train.txt"
    test_path = "../data/intention_test.txt"
    _split(sample_path, train_path, test_path)
    print("split sample to train & test done!\n")
