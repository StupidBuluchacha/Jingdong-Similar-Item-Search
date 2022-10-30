import jieba
import pandas as pd
import jieba.analyse
import jieba.posseg
import random

with open('../../Assignment1/datasets/product.txt', 'r') as f:
    my_data = f.read().split('\n')

def encode(words, fenci_product, fenci_adj, fenci_band):
    """
    按照词性，对word转为ner编码
    """
    # code映射为编码
    res = []
    for word in words:
        if word == None:
            continue
        result_list = []
        try:
            new_code = -1
            if word in fenci_product:  # 产品词
                new_code = "Product"
            elif word in fenci_adj:  # 修饰词
                new_code = "Adj"
            elif word in fenci_band:  # 品牌词
                new_code = "Band"
            # 对每个字符进行编码
            if new_code == -1:  # 其他词
                result_list = [x + '\tO' for x in word]
            else:
                result_list = [word[0] + '\tB-' + str(new_code)]
                if len(word) >= 1:
                    for i in range(1, len(word)):
                        result_list.append(word[i] + '\tI-' + str(new_code))
        except:
            #             print(word)
            continue

        res.extend(result_list)
    return res

def output_bio(results, file_name):
    """
    将bio格式的数组写入到file中
    """
    count = 0
    with open(file_name, 'w', encoding='utf8') as f:
        for word in results:
            for char in word:
                f.write(char + '\n')
            f.write('\n')
            count += 1
    print('已将{}个词写入到{}'.format(count, file_name))

def genTrainData(data):
    VALID_RATIO = 0.2
    TEST_RATIO = 0.2
    bio_result_list = []
    for index, row in data.iterrows():

        #     print(rows)
        fenci_product = row['cut_res'].split(',')
        fenci_adj = row['adj'].split(',')
        fenci_band = row['品牌'].split(',')
        querys = ','.join(jieba.cut(row['title'].strip())).split(',')
        temp_res = encode(querys, fenci_product, fenci_adj, fenci_band)
        if temp_res == None:
            continue
        bio_result_list.append(temp_res)
    random.shuffle(bio_result_list)
    valid_count = int(len(bio_result_list) * VALID_RATIO)
    test_count = int(len(bio_result_list) * TEST_RATIO)
    valid_list = bio_result_list[:valid_count]
    test_list = bio_result_list[valid_count:valid_count + test_count]
    train_list = bio_result_list[valid_count + test_count:]
    # 写入文件
    output_bio(train_list, '../datasets/train_NER.txt')
    output_bio(valid_list, '../datasets/valid_NER.txt')
    output_bio(test_list, '../datasets/test_NER.txt')

def dosegment_all(sentence):
    '''
    带词性标注，对句子进行分词，不排除停词等
    :param sentence:输入字符
    :return:
    '''

    sentence_seged = jieba.posseg.cut(sentence.strip())
    outstr = []
    for x in sentence_seged:
        if x.word in my_data:
            outstr.append(x.word)
    # 上面的for循环可以用python递推式构造生成器完成
    # outstr = ",".join([("%s/%s" %(x.word,x.flag)) for x in sentence_seged])
    return ','.join(outstr)

def dosegment_adj(sentence):
    '''
    带词性标注，对句子进行分词，不排除停词等
    :param sentence:输入字符
    :return:
    '''

    sentence_seged = jieba.posseg.cut(sentence.strip())
    outstr = []
    for x in sentence_seged:
        if x.flag in ['a', 'ad', 'ag', 'al', 'an']:
            outstr.append(x.word)
    #         outstr+="{}/{},".format(x.word,x.flag)
    #     上面的for循环可以用python递推式构造生成器完成
    # outstr = ",".join([("%s/%s" %(x.word,x.flag)) for x in sentence_seged])
    return ','.join(outstr)

test = pd.read_json("../../Assignment1/datasets/kb_jd_jsonl.txt",lines=True)

test['cut_res'] = test['title'].apply(dosegment_all)
test['adj'] = test['title'].apply(dosegment_adj)
print(test[['adj','cut_res','title','品牌']].head(30))
genTrainData(test)
