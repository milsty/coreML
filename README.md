# 苹果机器学习coreML框架实战：图像识别

有时候使用已有的模型无法满足你的需求，你想要自己创建模型，那么就需要按照下面这些步骤来做，这些步骤形成了机器学习流程，涵盖了从数据准备到模型构建、训练和最终部署的整个过程。下面将以识别香蕉图片的成熟度值为例子，讲解如何自建模型。
## 1. 定义问题
明确问题的类型是机器学习中非常关键的一步，因为不同类型的问题需要使用不同的算法和评估方法。我们可以问自己几个问题：
1. 输出是什么？
- 如果你的目标是预测一个连续的数值，比如销售额、温度等，那么这是一个回归问题。
- 如果你的目标是将样本分到不同的类别中，比如垃圾邮件分类、图像识别中的猫和狗，那么这是一个分类问题。
2. 数据标签：
- 查看数据集中的标签。如果标签是离散的（例如 A、B、C），那么这可能是一个分类问题。如果标签是连续的数值，那么这可能是一个回归问题。
3. 问题的业务背景：
- 了解问题的业务背景可以提供线索。例如，如果你在金融领域，并且任务是预测股票价格，这可能是一个回归问题。
通过综合考虑这些因素，你应该能够更清晰地确定问题的类型，从而为选择合适的算法和评估方法奠定基础。
以识别香蕉图片的成熟度值为例子，1.输出是香蕉的成熟度，这是一个连续的数值。2.数据集中的成熟度是一个连续的数值标签。因此我们可以根据这两个来确定这是一个回归问题。
## 2. 收集数据
收集与问题相关的数据。数据质量对模型性能至关重要，因此确保数据集包含足够的样本，并且这些样本是代表性的。寻找合适的数据集通常有以下途径：
1. 在线数据集平台： 许多在线平台提供各种各样的数据集，可供免费或付费下载。一些知名的平台包括 Kaggle、UCI Machine Learning Repository、Google Dataset Search 等。
2. 政府和组织网站： 政府机构和各种研究组织通常会发布公共数据集，这些数据集涵盖了各种领域。例如，美国统计局、欧洲开放数据平台等。
3. ...
本文例子中的数据集每一个item是香蕉图片，图片的名称是成熟度的评分。
[图片]
我们可以直接使用AnnotatedFiles来识别带标签的文件，读出来的格式就是：(文件,标签)。读出来后再调用mapFeatures(ImageReader.read)将非图片文件过滤掉。因为AnnotatedFiles读出来的标签格式是string类型，还需要调用mapAnnotations({ Float($0)! }，把评分从string转为浮点数float。最后调用flatMap把数组展平，之后更方便处理数据。
```
static let trainingDataURL = URL(fileURLWithPath: "~/Desktop/bananas")
// File name example: banana-5.jpg
let data = try AnnotatedFiles(labeledByNamesAt: trainingDataURL, separator: "-", index: 1, type: .image)
.mapFeatures(ImageReader.read)
.mapAnnotations({ Float($0)! })
.flatMap(augment)
```
## 3. 数据清理和预处理
对收集的数据进行清理和预处理。包括处理缺失值、处理异常值、标准化数据、进行特征工程等，以确保数据适合用于模型训练。如果是图片相关的模型，可以使用缩放、旋转、增加对比度来增强数据集。例子中采用对图片进行旋转和缩放。
AnnotatedFeature：特征和标签说为什么是这个格式
static func augment(_ original: AnnotatedFeature<CIImage, Float>) -> [AnnotatedFeature<CIImage, Float>] {
    let angle = CGFloat.random(in: -.pi ... .pi)
    let rotated = original.feature.transformed(by: .init(rotationAngle: angle))
    
    let scale = CGFloat.random(in: 0.8 ... 1.2)
    let scaled = original.feature.transformed(by: .init(scaleX: scale, y: scale))
    
    return [
        original,
        AnnotatedFeature(feature: rotated, annotation: original.annotation),
        AnnotatedFeature(feature: scaled, annotation: original.annotation),
    ]
}
苹果还提供图像显著性分析API，将图片的主体部分裁剪出来，以此强化图片主体。例如VNGenerateAttentionBasedSaliencyImageRequest。
## 4. 拆分数据集
将数据集分为训练集、验证集和测试集。训练集用于训练模型，验证集用于调整模型参数，测试集用于评估模型性能。我们将将数据集分为了两部分，80%是训练集，20%是测试集。
let (training, validation) = data.randomSplit(by: 0.8)
## 5. 选择模型
根据问题的性质选择适当的机器学习模型。不同的问题可能需要不同类型的模型，例如决策树、神经网络、支持向量机等。
本次是图像回归，那么需要两步：1. 提取图像特征。2.回归特征。回归特征的模型选择有多种，苹果提供学习模型，包括线性回归、决策树回归、随机森林等。我们先调用ImageFeaturePrint()方法作为特征提取器，再使用线性回归LinearRegressor()作为回归器。
let estimator = ImageFeaturePrint()
            .appending(LinearRegressor())
## 6. 训练模型
使用训练集来训练选择的模型。模型通过学习训练集中的模式和关系来调整其参数，以使其能够对未见过的数据做出准确的预测。
我们直接在fitted的方法里可以直接观察训练过程，也可以直接输出验证指标。如果这个模型的准确度不好，我们可以试图更改回归器的学习方法，增强数据集等方式来改进准确度。
```
let transformer = try await estimator.fitted(to: training, validateOn: validation) { event in
    guard let trainingMaxError = event.metrics[.trainingMaximumError] else {
        return
    }
    guard let validationMaxError = event.metrics[.validationMaximumError] else {
        return
    }
    print("Training max error: \(trainingMaxError), Validation max error: \(validationMaxError)")
}
```
例子中训练模型的过程是
[图片]
## 7. 评估模型
使用测试集来评估模型的性能。这可以通过各种指标如准确率、精确度、召回率、F1 分数等来完成，具体取决于问题的性质。苹果提供了多种损失函数来计算误差，例如最大绝对误差、平均绝对误差等。回归问题一般采用两种方法：
- 均方误差（Mean Squared Error，MSE）： 衡量模型预测值与实际值之间的平方差的平均值。目标是最小化MSE。
- 平均绝对误差（Mean Absolute Error，MAE）： 衡量模型预测值与实际值之间的绝对差的平均值。目标是最小化MAE。
```
func rootMeanSquaredError<T>([AnnotatedPrediction<T, T>]) -> T
func rootMeanSquaredError<T>(some Collection, some Collection) -> T
func maximumAbsoluteError<T>([AnnotatedPrediction<T, T>]) -> T
func maximumAbsoluteError<T>(some Collection, some Collection) -> T
func meanAbsoluteError<T>([AnnotatedPrediction<T, T>]) -> T
func meanAbsolutePercentageError<T>([AnnotatedPrediction<T, T>]) -> T
```
例子中使用平均绝对误差函数作为损失函数，API是meanAbsoluteError
```
let validationError = try await meanAbsoluteError(
    transformer.applied(to: validation.map(\.feature)),
    validation.map(\.annotation)
)
print("Mean absolute error: \(validationError)")
```
本模型的具体误差为：
## 8. 保存模型
一旦模型被训练和评估，可以将其部署到实际应用中，以进行实时的预测。coreML框架提供便捷的保存方法：调用write方法，直接将模型保存到定义的URL
```
try estimator.write(transformer, to: parametersURL)
```
公司内部访问可见飞书文档：https://bytedance.larkoffice.com/docx/RWLjduzv1oxs25xjDyLcp24Pn9d
