# Reproduction and Promotion of Single-cell RNA-seq denoising using a deep count  autoencoder

## Experiment Description

Denoise single-cell RNA-seq data. Single-cell RNA's RNA-seq is measured in biology experiments to analyse gene expression. Some RNA-seq cannot be accurately measured and is recorded as 0. However, there are also some genes actually do not express and their real value is 0. The work done in the original paper is using DCA to denoise single-cell RNA-seq data, that is, to judge whether the record 0 is generated from measurement or lack of expression, and to estimate the real value and fill in the appropariate position.

## Experiment Requirements

- 基本要求

  将test_data.csv,test_truedata.csv分为测试集和验证集。实现任意补插算法来对数据集data.csv进行补插。使用测试集确定一个较好的模型，并使用验证集验证。针对你的模型和实验写一份报告，并提交源码(说明运行环境)和可执行文件。(建议使用神经网络)

- 中级要求

  在test_data.csv上获得一定效果(dropout率变小，表达差异变小) 。在别人理论的基础上进行一些自己的修改。

- 高级要求

  对于data.csv的补插和生成data.csv的模拟真实数据集相比获得较好效果(dropout率变小，表达差异变小) 。

## Experiment Principle

![image-20250308204027742](/Users/haoxin/Library/Application Support/typora-user-images/image-20250308204027742.png)

Model Construction

```python
class AutoEncoder(nn.Module):
```

![image-20250308204114850](/Users/haoxin/Library/Application Support/typora-user-images/image-20250308204114850.png)

- Input: count matrix A

  $A_{ij}$ represents the expression the $i$ gene in the $j$ cell.

- Hidden Layer:

  $E, B, D$ in the picture above reprensent encoding layers, bottleneck layer and decoding layers.

  All hidden layers except the bottleneck layer includes 64 neurons, and the bottleneck layer has 32 neurons.

- Output: The three parameters of every gene $(μ, θ, π)$, that is (mean, dispersion, the probability of dropout), and is also the three parameters of ZINB distribution.

#### Specify the numbers of neurons per layer

```python
def __init__(self, input_size=None, hasBN=False):
        """
        该autoencoder还可以改进：
        dropout层

        :param input_size:
        """
        super().__init__()
        self.intput_size = input_size
        self.hasBN=hasBN
        self.input = Linear(input_size, 64)
        self.bn1 = BatchNorm1d(64)
        self.encode = Linear(64, 32)
        self.bn2 = BatchNorm1d(32)
        self.decode = Linear(32, 64)
        self.bn3 = BatchNorm1d(64)
        self.out_put_PI = Linear(64, input_size)
        self.out_put_M = Linear(64, input_size)
        self.out_put_THETA = Linear(64, input_size)
        self.reLu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
```

#### Specify the activation function

The activation function of THETA and M are both exp, this can make sure the output is legitimate because they are both non-negative values.

Use Sigmoid as the activation function, because it is the estimated dropout probability and it should be between 0 and 1.

```python
def forward(self, x):
        x = self.input(x)
        if self.hasBN:x = self.bn1(x)
        x = self.reLu(x)
        x = self.encode(x)
        if self.hasBN:x = self.bn2(x)
        x = self.reLu(x)
        x = self.decode(x)
        if self.hasBN:x = self.bn3(x)
        x = self.reLu(x)
        PI = self.sigmoid(self.out_put_PI(x))
        M = torch.exp(self.out_put_M(x))
        THETA = torch.exp(self.out_put_THETA(x))
        return PI, M, THETA
```

#### Define the loss function of the model 

![image-20250308213147688](/Users/haoxin/Library/Application Support/typora-user-images/image-20250308213147688.png)

NLL is the negative logarithmic likelihood, and the ZINB formula is as below:

![image-20250308213444761](/Users/haoxin/Library/Application Support/typora-user-images/image-20250308213444761.png)

The ZINB distribution is chosen is because scRNA-seq data has many 0 values, and whose negative logarithmic likelihood is used as the loss function. When the loss function is minimum, the probability and the liklihood of ZINB are greatest. The training to be conducted is to reduce the loss and get the best estimation of ZINB parameters. Then use the estimated ZINB function to generate interpolation values.

Loss function definiton:

```python
eps = torch.tensor(1e-10)
        THETA = torch.minimum(THETA, torch.tensor(1e6))
        t1 = torch.lgamma(THETA + eps) + torch.lgamma(X + 1.0) - torch.lgamma(X + THETA + eps)
        t2 = (THETA + X) * torch.log(1.0 + (M / (THETA + eps))) + (X * (torch.log(THETA + eps) - torch.log(M + eps)))
        nb = t1 + t2
        nb = torch.where(torch.isnan(nb), torch.zeros_like(nb) + max1, nb)
        nb_case = nb - torch.log(1.0 - PI + eps)
        zero_nb = torch.pow(THETA / (THETA + M + eps), THETA)
        zero_case = -torch.log(PI + ((1.0 - PI) * zero_nb) + eps)
        res = torch.where(torch.less(X, 1e-8), zero_case, nb_case)
        res = torch.where(torch.isnan(res), torch.zeros_like(res) + max1, res)
        return torch.mean(res)
```

#### Input

The input function is as shown below. $X$ is original counting matrix, and $S_i$ is the factor of proportion of every cell.

$$\overline{X} = zscore(log(diag(s_i)^{-1}X + 1))$$

Because NLL estimation has been used, we need to preprocess infinite and invalid values in original dataset. Implement z-score normalization, that is, subtract the mean from the data and divide it by its standard deviation. The outcome of the normalization is that for every attribute, all data is clustered around 0 and is with a deviation of 1. The formula is: $x^*=\frac{x - \overline{x}}{\sigma}$

```python 
def preprocess_data(data: Tensor):
    gene_num = data.shape[1]
    s = torch.kthvalue(data,gene_num//2,1)
    s = 1/s.values
    norm_data = torch.matmul(torch.diag(s), data) + 1
    norm_data = torch.log(norm_data)
    norm_data = (norm_data - norm_data.mean()) / norm_data.std()
    return norm_data
```

#### Training

Use autoencoder for training, use batchsize = 32, that is, use the data of 32 cells for training each time.

```python
def train(EPOCH_NUM=100, print_batchloss=False, autoencoder=None, loader=None, startEpoch=0):
    """

    :param print_batchloss: if print train infor, False by default
    """
    opt = Adam(autoencoder.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY)
    # opt = SGD(autoencoder.parameters(), lr=1e-2, momentum=0.8)
    mean_loss=0
    for epoch in range(EPOCH_NUM+1):
        epoch_loss = 0

        for batch, batch_data in enumerate(loader):
            # batchsize = 32

            opt.zero_grad()
            train_batch = batch_data[0]
           
            # d: original matrix
            d = train_batch[:, :, 0]
            # norm_d: proessed data
            norm_d = train_batch[:, :, 1]
```

Forward and calculate loss value. 

Follow the sequence of input to output, calculate and store intermediate variables of the model to train the next layer.

```python
# forward
            PI, M, THETA = autoencoder(norm_d)
            templ = lzinbloss(d, PI, M, THETA)
            epoch_loss += templ
            if print_batchloss:
                print(f'epoch:{epoch+startEpoch},batch:{batch},batch loss:{templ},(batch size {BATCHSIZE})')
                f.write(f'epoch:{epoch+startEpoch},batch:{batch},batch loss:{templ},(batch size {BATCHSIZE})\n')
           
            # gradient descent
            opt.step()
        mean_loss+=epoch_loss
        if epoch % 100 ==0:
            mean_loss=mean_loss/100
            print(f'epoch:{epoch+startEpoch},epoch loss:{mean_loss}')
            f.write(f'epoch:{epoch+startEpoch},epoch loss:{mean_loss}\n')
            mean_loss=0
        # if epoch % EVER_SAVING == 0 and epoch!=0: torch.save(autoencoder.state_dict(), open(f'0113epoch{epoch+startEpoch}.pkl', 'wb'))
        if epoch % EVER_SAVING == 0 and epoch!= 0 : torch.save(autoencoder.state_dict(), open(f'0113epoch{epoch+startEpoch}withoutBN.pkl', 'wb'))
```

Compute gradients with backpropagation

```python
						templ.backward()
            clip_grad_norm_(autoencoder.parameters(), max_norm=5, norm_type=2)
```

Considering the output PI,M,THETA, if the gene's corresponding PI>0.5, fill with corresponding M value as below:

```python
 autoencoder.load_state_dict(torch.load(STATE_DICT_FILE))
    print(autoencoder)
    PI, M, THETA = autoencoder(norm_data)
    iszero = data == 0
    predict_dropout_of_all = PI>0.5
    # dropout_predict = torch.where(predict_mask, M, torch.zeros_like(PI))
    # print("after",after)

    true_drop_out_mask = iszero*((truedata - data)!=0)
    predict_dropout_mask = iszero*predict_dropout_of_all
    after = torch.floor(torch.where(predict_dropout_mask,M,data))
    zero_num = iszero.sum()
    true_dropout_num = true_drop_out_mask.sum()
    predict_dropout_num = predict_dropout_mask.sum()
    print("predict_dropout_num:",predict_dropout_num,
          "\ntrue_dropout_num:", true_dropout_num,
          "\nzero_num:",zero_num,
          "\npredict out of true dropout rate:",(predict_dropout_mask*true_drop_out_mask).sum()/true_dropout_num)

    dif_after =  truedata - after
    dif_true = truedata - data
    # print(dif_after)
    # print(dif_true)
    print("predict distance:", torch.sqrt(torch.square(truedata - after).sum()).data,
          "origin distance:", torch.sqrt(torch.square(truedata - data).sum()).data)
```

## Analysis and Result Display

Split *test_data.csv,test_truedata.csv* into test set and validation set in a ratio of 7:3.

### Test four models on test dataset

hasBN: whether has BatchNorm1d layer

zinb_new: whether use the improved zinb function

##### Model I：hasBN=False zinb_new=False

Gradient explosion occurred in the later stage of training.

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250308230742214.png" alt="image-20250308230742214" style="zoom:50%;" />

1000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250312233017458.png" alt="image-20250312233017458" style="zoom: 25%;" />

Use Euclidean distance to represent the difference of predict value and real value of data. The difference is tremendous, and it is clearly underfitting.

2000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250312233520468.png" alt="image-20250312233520468" style="zoom: 25%;" />

Still underfitting.

3000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309095929454.png" alt="image-20250309095929454" style="zoom:50%;" />

Gradient explosion.

##### Model II：hasBN=True zinb_new=False

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100032757.png" alt="image-20250309100032757" style="zoom:50%;" />

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100048875.png" alt="image-20250309100048875" style="zoom:50%;" />

1000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100122029.png" alt="image-20250309100122029" style="zoom: 50%;" />

2000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100205094.png" alt="image-20250309100205094" style="zoom: 50%;" />

3000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100228528.png" alt="image-20250309100228528" style="zoom: 50%;" />

##### Model III：hasBN=False zinb_new=True

3000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100333174.png" alt="image-20250309100333174" style="zoom: 50%;" />

##### Model IV：hasBN=False zinb_new=True

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100404073.png" alt="image-20250309100404073" style="zoom: 50%;" />

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100411401.png" alt="image-20250309100411401" style="zoom: 50%;" />

1000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100434454.png" alt="image-20250309100434454" style="zoom:67%;" />

2000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100452001.png" alt="image-20250309100452001" style="zoom:67%;" />

3000 epochs:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309100511985.png" alt="image-20250309100511985" style="zoom:67%;" />

Conclusion: Model IV has the best performance. The new zinb loss function is better, and using batchNorm1d layer is better than not using it.

### Validate on the validation set

Predict on the validation set using 3000 epoch model.

Model I:

Model I has poor performance, and gradient explode after 3000 epoches, so it is not comparable. The outcome is not listed here.

Model II:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309193510108.png" alt="image-20250309193510108" style="zoom:67%;" />

Model III:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309193631122.png" alt="image-20250309193631122" style="zoom:67%;" />

Model IV:

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309193658078.png" alt="image-20250309193658078" style="zoom:67%;" />

It can be seen that model IV has the best performance, with high precision and recall, and the lowest predict distance. 

### Analyzation

Model I is the original model reproduced as described in the paper. However, the performance of Model I is undesirable, and the speed of its gradient descent is slow. Besides, after certain epoches, the loss is NA, which represents calculation error or numerical anormaly. This can be caused by gradient explosion, which makes the parameters to be extreme and leads to overflow, making the value be inf, and result in NaN eventually. Thus the batch normalization layer is introduced into the neural network, that is, set `autoencoder = AutoEncoder(1000,hasBN=True) ` , and test its performance. It can be seen that the denoise of data.csv has a fair performance compared with the simulated real dataset that generated data.csv, with less dropout and less difference.

After reading the whole paper, it is found that some experiment result cannot be reproduced according to the method described in the paper. Besides, the source code is different with the text description. Thus, re-implement loss calculation and training technique as below:

 ```python
class LZINBLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X: Tensor, PI: Tensor = None, M: Tensor = None, THETA: Tensor = None):
        # in case of log(0) log (neg num) 
        eps = self.eps
        # deal with inf
        max1 = max(THETA.max(), M.max())
        if THETA.isinf().sum() != 0:
            THETA = torch.where(THETA.isinf(), torch.full_like(THETA, max1), THETA)
        if M.isinf().sum() != 0:
            M = torch.where(M.isinf(), torch.full_like(M, max1), M)


        if PI.isnan().sum() != 0:
            PI = torch.where(PI.isnan(), torch.full_like(PI, eps), PI)
        if THETA.isnan().sum() != 0:
            THETA = torch.where(THETA.isnan(), torch.full_like(THETA, eps), THETA)
        if M.isnan().sum() != 0:
            M = torch.where(M.isnan(), torch.full_like(M, eps), M)

        eps = torch.tensor(1e-10)
        THETA = torch.minimum(THETA, torch.tensor(1e6))
        t1 = torch.lgamma(THETA + eps) + torch.lgamma(X + 1.0) - torch.lgamma(X + THETA + eps)
        t2 = (THETA + X) * torch.log(1.0 + (M / (THETA + eps))) + (X * (torch.log(THETA + eps) - torch.log(M + eps)))
        nb = t1 + t2
        nb = torch.where(torch.isnan(nb), torch.zeros_like(nb) + max1, nb)
        nb_case = nb - torch.log(1.0 - PI + eps)
        zero_nb = torch.pow(THETA / (THETA + M + eps), THETA)
        zero_case = -torch.log(PI + ((1.0 - PI) * zero_nb) + eps)
        res = torch.where(torch.less(X, 1e-8), zero_case, nb_case)
        res = torch.where(torch.isnan(res), torch.zeros_like(res) + max1, res)
        return torch.mean(res)
 ```

```python
def train(EPOCH_NUM=100, print_batchloss=False, autoencoder=None, loader=None, startEpoch=0):
    opt = Adam(autoencoder.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY)
    # opt = SGD(autoencoder.parameters(), lr=1e-2, momentum=0.8)
    mean_loss=0
    for epoch in range(EPOCH_NUM+1):
        epoch_loss = 0

        for batch, batch_data in enumerate(loader):
            opt.zero_grad()
            train_batch = batch_data[0]
            d = train_batch
            PI, M, THETA = autoencoder(d)
            templ = lzinbloss(d, PI, M, THETA)
            epoch_loss += templ
            if print_batchloss:
                print(f'epoch:{epoch+startEpoch},batch:{batch},batch loss:{templ},(batch size {BATCHSIZE})')
                f.write(f'epoch:{epoch+startEpoch},batch:{batch},batch loss:{templ},(batch size {BATCHSIZE})\n')
            # backpropagtion
            templ.backward()
            clip_grad_norm_(autoencoder.parameters(), max_norm=5, norm_type=2)
            # gra descent
            opt.step()
        print(f'epoch:{epoch+startEpoch},epoch loss:{epoch_loss}')
        f.write(f'epoch:{epoch+startEpoch},epoch loss:{mean_loss}\n')
        # mean_loss=0
        if epoch % EVER_SAVING == 0 and epoch!=0: torch.save(autoencoder.state_dict(), open(f'0113epoch{epoch+startEpoch}.pkl', 'wb'))
        # if epoch % EVER_SAVING == 0 and epoch!= 0 : torch.save(autoencoder.state_dict(), open(f'0113epoch{epoch+startEpoch}withoutBN.pkl', 'wb'))

```

After the improvement, test on the test set, the result is shown as below, and the denoised data is saved in result.csv.

<img src="/Users/haoxin/Library/Application Support/typora-user-images/image-20250309205101055.png" alt="image-20250309205101055" style="zoom: 25%;" />

![image-20250309205130717](/Users/haoxin/Library/Application Support/typora-user-images/image-20250309205130717.png)
