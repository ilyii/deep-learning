# Deep Learning

Welcome to the Deep Learning project repository! This collection showcases a variety of deep learning applications and side projects. Each directory in this repository represents a different application or research area in the field of deep learning. Below is an overview of the projects included in this repository.

## Content

### 1. [Emotion Recognition](emotion_recognition)
A project focused on recognizing emotions from textual or visual inputs. It includes two sub-projects:
- **[BERT-Based](emotion_recognition/bert)**: Utilizes BERT (Bidirectional Encoder Representations from Transformers) for text-based emotion recognition.
- **[RNN-Based](emotion_recognition/rnn)**: Employs Recurrent Neural Networks (RNNs) for emotion detection, suitable for sequential data.

### 2. [Image Classification](image_classfication)
Projects that tackle various image classification problems using different architectures:
- **[CNN on CIFAR](image_classfication/cnn_cifar)**: Implements Convolutional Neural Networks (CNNs) to classify images from the CIFAR dataset.
- **[NN on MNIST](image_classfication/nn-mnist)**: Uses a basic Neural Network to classify handwritten digits from the MNIST dataset.

### 3. [Titanic](titanic)
An application that applies machine learning techniques to predict survival outcomes for passengers on the Titanic based on features such as age, gender, and class.

### 4. [Transfer Learning](transfer-learning)
Explores transfer learning techniques where a pre-trained model is adapted for a new, but related task. This directory includes various scripts for model training and evaluation.

## File Structure

- **config.py**: Configuration settings for the respective project.
- **data.py**: Data preprocessing and handling scripts.
- **main.py**: Entry point for running the main application or experiment.
- **model.py**: Definition of the neural network architecture.
- **train.py**: Scripts for training the models.
- **test.py** / **eval.py**: Scripts for testing and evaluating the models.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd deep-learning
   ```
3. **Install Dependencies**:
   Each project may have its own set of dependencies.

4. **Run the Project**:
   Each project has a `main.py` script that serves as the entry point. You can run the scripts using Python:
   ```bash
   python <project-directory>/main.py
   ```
