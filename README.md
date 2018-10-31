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
