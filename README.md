# TianChi_OGeek
## 天池数据挖掘竞赛，预测广告点击-一个二分类问题
### 2018.10.31
* 根据label的0-1将数据进行切分，positive对应label==1，negative对应label==0
* 由于数据量有200万行，用for循环一行一行分会消耗大量时间，所以采用df.isin()进行
  * 具体操作如：
  <pre><code>positive = data[data['label'].isin([1])]
  negative = data[data['label'].isin([0])]</code></pre>
  该函数将data.label = 0的样本放到positive里面， 将data.label = 1的样本放到negative里面
  * 得到结果positive.shape[0] = 732256, negative.shape[0] = 1234975
### 2018.11.5
* 利用word2vec模块，将'title','prefix'以及'query_predict'的预测查询词条提取后输入word2vec模型训练求得词与词之间的距离  
即数值化query_predict属性为了之后训练分类模型做准备(提升样本向量的维度)具体代码如下：
<pre><code>#将提取的所有词条存入‘all_words.txt’文件中
from gensim.models import word2vec
from gensim.models import Word2Vec
#注意这两个模块不一样的

s = word2vec.Text8Corpus('.../all_words.txt')
model = Word2vec(s,size = 200, min_count = 1)
#其中size参数是词条转换成向量的维度，维度越大之后的分类效果越好，然而计算词向量的时间也越长
#min_count参数是将s中出现次数为min_count以上的词条进行计算词向量</code></pre>
至此word2vec模型训练完毕，可以通过model.similarity(a,b)来计算s中两个词条a,b之间词向量的距离
* 利用df.sample(frac)函数对样本集进行采样，并分别计算prefix与各个query_predict之间的词向量距离(这一步较耗时)  
在计算词向量距离是，利用df.apply()函数配合lambda函数来提高效率，具体代码如下：
<pre><code>#由于原单个词条中会出现空格，导致部分带空格的词条在转换成vector时被分成多个词条，
#在进行计算词向量距离的时候无法在巨量苦衷找到带空格的词条所对应的词向量，
#在这边我们自定义一个可以跳过带空格词条的计算距离函数judge()
def judge(rowa, rowb, l)
"""
计算rowa，rowb两列词条词向量的距离，这里的l是语料库
"""
    if rowa not in l or rowb not in l:
        return 0
    else:
        return model.similarity(rowa, rowb)

p_q_0 = data.apply(lambda row: judge(data['prefix'],data['query_key_0'], l), axis = 1)</code></pre>
* baseline利用统计，将tag下的20余种分类进行统计，计算每个类型的点击率，并用该点击率代替tag属性，最终分类效果并不理想
之后考虑到类别属性常用的处理方法是进行度热编码，于是在之后的模型训练中用到的都是tag的独热编码属性，为此了解了下pandas的get_dummies()函数，
处理示例如下：
<pre><code>D = pd.get_dummies(data,columns = ['tag'])
#返回的是将tag度热编码后的DataFrame，列名为'tag_0'~'tag_20'</code></pre>
* A榜最好成绩是利用42维属性，10万样本，LightGBM算法训练得到的分类模型，阈值为0.42，本地F1值为0.7，线上F1值为0.5850
