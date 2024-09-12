# 🎯 Neural Network Regularization with L2 and Dropout

Welcome to this project on **Regularization** techniques in neural networks! In this repository, we implement and explore two key regularization methods: **L2 Regularization** and **Dropout** to improve model generalization and performance. Below, you'll find a detailed explanation of the code, along with its key components and results.

Let's dive into the project!

## 🧠 Introduction to Regularization

Regularization is essential for improving the generalization ability of machine learning models. It helps prevent **overfitting**, ensuring that the model performs well not only on the training data but also on unseen test data. In this project, we focus on:

- **L2 Regularization:** Adds a penalty proportional to the squared value of the weights, which discourages large weight values.
- **Dropout Regularization:** Randomly turns off a fraction of neurons during training to prevent the network from becoming too reliant on specific neurons.

## 📂 Project Structure

The following key files and libraries are used in this repository:

- `reg_utils.py`: Contains utility functions such as `sigmoid`, `relu`, and `initialize_parameters`.
- `testCases.py`: Includes test cases to verify the correctness of the functions.
- `sklearn.datasets`: Generates datasets for training and testing.
- `matplotlib`: For plotting decision boundaries and cost functions.

## 📊 Regularization Techniques

### 1️⃣ Non-Regularized Model

The `model` function implements a 3-layer neural network:

- **Architecture**: `LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID`
- **Cost Function**: Standard cross-entropy loss.

```python
parameters = model(train_X, train_Y)
```

Results for the non-regularized model:

- **Training Accuracy**: 94.79%
- **Test Accuracy**: 91.5%

#### 🖼️ Plotting the Decision Boundary

The decision boundary of the non-regularized model shows that it fits the training data well, but there's room for improvement in generalization.

```python
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

### 2️⃣ L2 Regularization

L2 regularization is applied by adding the squared weights to the cost function. The function `compute_cost_with_regularization` computes the regularized cost:

```python
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    L2_regularization_cost = (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost
```

The regularized model is trained with `lambda = 0.7`:

```python
parameters = model(train_X, train_Y, lambd=0.7)
```

Results for the L2-regularized model:

- **Training Accuracy**: 93.5%
- **Test Accuracy**: 94.0%

### 3️⃣ Dropout Regularization

Dropout is implemented by randomly deactivating a fraction of the neurons during training. The `forward_propagation_with_dropout` function applies dropout regularization:

```python
def forward_propagation_with_dropout(X, parameters, keep_prob):
    # Randomly shut down neurons during training with probability 1-keep_prob
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob
    A1 = np.multiply(A1, D1)
    A1 /= keep_prob
    return A3, cache
```

The model is trained with `keep_prob = 0.86`:

```python
parameters = model(train_X, train_Y, keep_prob=0.86)
```

Results for the dropout-regularized model:

- **Training Accuracy**: 92.0%
- **Test Accuracy**: 93.5%

## 🔍 Evaluation

By using both L2 and Dropout regularization, we improved the test set accuracy and generalization of the neural network. Here's a comparison of the performance:

| Model Type            | Training Accuracy | Test Accuracy |
|-----------------------|-------------------|---------------|
| Non-Regularized       | 94.79%            | 91.5%         |
| L2 Regularization     | 93.5%             | 94.0%         |
| Dropout Regularization| 92.0%             | 93.5%         |

## 🔗 Acknowledgements

This project was developed as part of the **Deep Learning Specialization** by [DeepLearning.AI](https://www.deeplearning.ai/courses/deep-learning-specialization/). Special thanks to their incredible team for providing the foundational content.

Happy coding! 😊🎉
