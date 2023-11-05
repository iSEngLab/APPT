from transformers import BertModel, AdamW, BertTokenizer
import torch

class Config(object):
    '''
    配置参数
    '''
    def __init__(self):
        # 数据路径
        self.data_train_path = './dataset/Small/data_code_train_'
        self.data_test_path = './dataset/Small/data_code_test_'
        # 模型保存路径
        self.model_save_path = './output/bert/small/add/head'

        # 模型测试路径
        self.data_test = '../output'
        self.model_test_path = ''

        # 针对长度超过bert限制的buggy和fixed代码的截断方式
        self.cutMethod = 'headTail'
        # self.cutMethod = 'head'
        # self.cutMethod = 'tail'
        # self.cutMethod = 'mid'

        # buggy与fixed经过bert生成embedding的拼接方式
        self.splicingMethod = 'cat'
        # self.splicingMethod = 'add'
        # self.splicingMethod = 'sub'
        # self.splicingMethod = 'mul'
        # self.splicingMethod = 'mix'

        # GPU 配置使用检测
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_ids = [0,1]
        # GPU 是否使用cuda
        self.use_cuda = True

        # bert 预训练模型
        self.model_path = 'bert-base-uncased'
#        self.model_path = 'microsoft/codebert-base'
#        self.model_path = 'microsoft/graphcodebert-base'
        # bert 是否冻结
        self.freeze_bert = False
        # 模型的最长输入，bert为512，longformer为4096
#        self.max_length = 4096
        self.max_length = 512

        # lstm 输入数据特征维度：Bert模型 token的embedding维度 = Bert模型后接自定义分类器（单隐层全连接网络）的输入维度
        self.input_size = 768
        # lstm 隐层维度
        self.hidden_size = 256
        # lstm 循环神经网络层数
        self.num_layers = 2
        # dropout：按一定概率随机将神经网络单元暂时丢弃，可以有效防止过拟合
        self.dropout = 0.5

        # linear 输入特征size
        self.num_classes = 1

        # epoch 整体训练次数
        self.num_epoch = 50
        # epoch 开始训练时已处于第几次，默认为0
        self.start_epoch = 0
        # batch 训练batch大小
        self.train_batch_size = 16
        # batch 测试batch大小
        self.test_batch_size = 1

        self.run_rq3 = False
