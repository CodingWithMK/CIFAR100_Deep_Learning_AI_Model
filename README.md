### README (English)

# Unsupervised Machine Learning on CIFAR-100 Dataset

## Project Overview

This project focuses on applying unsupervised machine learning techniques on the CIFAR-100 dataset. Two different approaches were implemented: 

1. **K-Means Clustering with PCA**: The data was first reduced using Principal Component Analysis (PCA) before applying the K-Means clustering algorithm for grouping similar images.
2. **Autoencoder Neural Network**: A simple encoder-decoder architecture was trained to reconstruct images using an autoencoder.

Both approaches aim to analyze and cluster the dataset without any labeled data. Exploratory Data Analysis (EDA) was conducted to better understand the dataset, followed by model implementation and performance evaluation.

## Dataset

- **CIFAR-100**: A dataset containing 60,000 32x32 color images across 100 classes, with 600 images per class.
- Data was split into training and test sets as follows:
  - **X_train**: Training data
  - **X_test**: Test data

Since this is an unsupervised learning task, no separate validation sets were created.

## Tools and Libraries

- **TensorFlow**
- **Pandas**
- **NumPy**
- **Scikit-learn**

These libraries were used for data handling, model building, and evaluation.

## Project Steps

1. **Data Preprocessing**: The dataset was loaded and split into training and test sets. Images were normalized to `float32` type for easier processing.
2. **Exploratory Data Analysis (EDA)**: Randomly selected 100 images from the training set were visualized, and the pixel and color structures of the first image were analyzed in detail. 
3. **Flattening**: The images were flattened for further model training.
4. **Model 1: K-Means Clustering**:
   - Two methods were tested:
     1. Directly applying K-Means clustering to the dataset.
     2. First applying PCA for dimensionality reduction, followed by K-Means clustering.
   - Predictions were visualized using scatter plots.
5. **Model 2: Autoencoder Neural Network**:
   - A basic autoencoder was constructed with a simple encoder-decoder architecture.
   - After training, the loss and validation loss metrics were visualized.
   - The network reconstructed 10 images from the test set, and the original vs reconstructed images were visualized.

## Visualizations

- **Scatter Plot**: For visualizing the results of K-Means clustering with and without PCA.
- **Reconstructed Images**: Comparison of original and reconstructed images by the Autoencoder.

---

### README (Türkçe)

# CIFAR-100 Veri Seti Üzerinde Gözetimsiz Makine Öğrenmesi

## Proje Özeti

Bu projede, CIFAR-100 veri seti üzerinde gözetimsiz makine öğrenmesi teknikleri uygulanmıştır. İki farklı yöntem kullanılmıştır:

1. **PCA ile K-Means Kümeleme**: K-Means kümeleme algoritması uygulanmadan önce, veriler Başlıca Bileşen Analizi (PCA) yöntemi ile boyut indirgeme işlemine tabi tutulmuştur.
2. **Autoencoder Sinir Ağı**: Basit bir encoder-decoder mimarisi kullanılarak bir autoencoder eğitilmiştir ve bu model, görselleri yeniden inşa etmek için kullanılmıştır.

Her iki yaklaşım da etiketli veriler olmadan veri setini analiz etmeyi ve kümelemeyi hedeflemiştir. Projenin başında veri setini daha iyi anlamak için Keşifsel Veri Analizi (EDA) gerçekleştirilmiş, ardından model oluşturma ve performans değerlendirmesi yapılmıştır.

## Veri Seti

- **CIFAR-100**: 100 sınıf içeren, her bir sınıfta 600 adet bulunan 32x32 boyutlarında 60,000 renkli görüntü içerir.
- Veri eğitim ve test kümesi olarak ayrılmıştır:
  - **X_train**: Eğitim verisi
  - **X_test**: Test verisi

Bu proje gözetimsiz öğrenme üzerine olduğundan, ayrı bir doğrulama kümesi oluşturulmamıştır.

## Araçlar ve Kütüphaneler

- **TensorFlow**
- **Pandas**
- **NumPy**
- **Scikit-learn**

Bu kütüphaneler, veri işleme, model oluşturma ve değerlendirme süreçlerinde kullanılmıştır.

## Proje Adımları

1. **Veri Ön İşleme**: Veri seti yüklendi ve eğitim ve test kümelerine ayrıldı. Görseller `float32` veri türüne normalize edildi.
2. **Keşifsel Veri Analizi (EDA)**: Eğitim kümesinden rastgele seçilen 100 adet görüntü görselleştirildi ve ilk görüntünün piksel ve renk yapıları detaylı olarak analiz edildi.
3. **Düzleştirme (Flattening)**: Görseller, model eğitimi için düzleştirildi.
4. **Model 1: K-Means Kümeleme**:
   - İki farklı yöntem denendi:
     1. K-Means kümeleme algoritmasının doğrudan veri setine uygulanması.
     2. PCA ile boyut indirgeme yapıldıktan sonra K-Means kümeleme algoritmasının uygulanması.
   - Tahmin sonuçları scatter plot kullanılarak görselleştirildi.
5. **Model 2: Autoencoder Sinir Ağı**:
   - Basit bir encoder-decoder mimarisi ile bir autoencoder oluşturuldu.
   - Eğitim sonrası loss ve validation loss metrikleri görselleştirildi.
   - Sinir ağı, test kümesindeki 10 adet görüntüyü yeniden yapılandırdı ve orijinal görüntüler ile yeniden yapılandırılmış görüntüler karşılaştırılarak görselleştirildi.

## Görselleştirmeler

- **Scatter Plot**: PCA ile ve PCA olmadan K-Means kümeleme sonuçlarının görselleştirilmesi.
- **Yeniden Yapılandırılmış Görseller**: Autoencoder tarafından orijinal ve yeniden yapılandırılmış görsellerin karşılaştırılması.

---




### README (English)

# Supervised Machine Learning on CIFAR-100 Dataset

## Project Overview

This project implements supervised machine learning techniques on the CIFAR-100 dataset using both traditional machine learning classifiers and a Convolutional Neural Network (CNN). Two approaches were tested:

1. **Traditional Machine Learning Models**: Data was scaled using StandardScaler, and three classifiers were trained: RandomForestClassifier, K-Nearest Neighbors (KNN), and DecisionTreeClassifier. Model performance was evaluated using accuracy score, F1 score, precision, and recall metrics.
2. **Convolutional Neural Network (CNN)**: A CNN architecture was designed and trained for image classification. The network was then used to predict the labels of test images.

## Dataset

- **CIFAR-100**: A dataset consisting of 60,000 32x32 color images from 100 classes, with 600 images per class.
- Data was split into:
  - **X_train, y_train**: Training data
  - **X_test, y_test**: Test data

## Tools and Libraries

- **TensorFlow**
- **Pandas**
- **NumPy**
- **Scikit-learn**

These libraries were used for data manipulation, model training, and evaluation.

## Project Steps

1. **Data Preprocessing**: 
   - The dataset was loaded and split into training and test sets.
   - Normalization was applied to both `X_train` and `X_test` by converting the data to `float32` format.
   - Exploratory Data Analysis (EDA) was performed, where 100 random images from the training set were visualized, and pixel and color structures of the first image were analyzed.
   - Flattening of the images was done before passing the data to the machine learning models.

2. **Model 1: Traditional Classifiers**:
   - **Scaling**: Data was scaled using `StandardScaler` to standardize the input.
   - **Training**: Three models were trained:
     1. **RandomForestClassifier**
     2. **K-Nearest Neighbors (KNN)**
     3. **DecisionTreeClassifier**
   - **Evaluation**: The models were evaluated using `accuracy_score`, `F1 score`, `precision`, and `recall`.

3. **Model 2: Convolutional Neural Network (CNN)**:
   - A simple CNN architecture was built and trained on the CIFAR-100 dataset.
   - The CNN was tested on the test set, where predictions were made on the images in `y_test`.

4. **Visualization and Evaluation**:
   - Model performance was visualized using accuracy, precision, recall, and F1 scores for each classifier.
   - For the CNN, the network’s predictions were compared with the true labels, and classification accuracy was reported.

## Future Work

This project demonstrates the use of both traditional machine learning classifiers and CNNs for image classification on the CIFAR-100 dataset. Future improvements could include testing more complex CNN architectures or exploring hyperparameter tuning for traditional classifiers.

---

### README (Türkçe)

# CIFAR-100 Veri Seti Üzerinde Gözetimli Makine Öğrenmesi

## Proje Özeti

Bu projede, CIFAR-100 veri seti üzerinde gözetimli makine öğrenmesi teknikleri uygulanmıştır. Hem geleneksel makine öğrenmesi sınıflandırıcıları hem de bir Evrişimli Sinir Ağı (CNN) kullanılarak iki farklı yöntem test edilmiştir:

1. **Geleneksel Makine Öğrenmesi Modelleri**: Veriler StandardScaler kullanılarak ölçeklendirilmiş ve üç sınıflandırıcı eğitilmiştir: RandomForestClassifier, K-Nearest Neighbors (KNN), ve DecisionTreeClassifier. Modeller, accuracy score, F1 score, precision ve recall metrikleri ile değerlendirilmiştir.
2. **Evrişimli Sinir Ağı (CNN)**: Görüntü sınıflandırması için bir CNN mimarisi oluşturulmuş ve eğitilmiştir. Sinir ağı, test verisindeki görüntüler üzerinde tahmin yapmak için kullanılmıştır.

## Veri Seti

- **CIFAR-100**: 100 sınıf içeren, her bir sınıfta 600 adet bulunan 32x32 boyutlarında 60,000 renkli görüntü içerir.
- Veri eğitim ve test kümesi olarak ayrılmıştır:
  - **X_train, y_train**: Eğitim verisi
  - **X_test, y_test**: Test verisi

## Araçlar ve Kütüphaneler

- **TensorFlow**
- **Pandas**
- **NumPy**
- **Scikit-learn**

Bu kütüphaneler, veri işleme, model eğitimi ve değerlendirme süreçlerinde kullanılmıştır.

## Proje Adımları

1. **Veri Ön İşleme**:
   - Veri seti yüklendi ve eğitim ile test kümelerine ayrıldı.
   - Hem `X_train` hem de `X_test` verileri `float32` formatına dönüştürülerek normalize edildi.
   - Keşifsel Veri Analizi (EDA) yapıldı; eğitim kümesinden rastgele seçilen 100 adet görüntü görselleştirildi ve ilk görüntünün piksel ve renk yapıları analiz edildi.
   - Görseller makine öğrenmesi modellerine iletilmeden önce düzleştirildi (flattening).

2. **Model 1: Geleneksel Sınıflandırıcılar**:
   - **Ölçeklendirme**: `StandardScaler` kullanılarak veri standartlaştırıldı.
   - **Eğitim**: Üç model eğitildi:
     1. **RandomForestClassifier**
     2. **K-Nearest Neighbors (KNN)**
     3. **DecisionTreeClassifier**
   - **Değerlendirme**: Modeller accuracy score, F1 score, precision ve recall metrikleri ile değerlendirildi.

3. **Model 2: Evrişimli Sinir Ağı (CNN)**:
   - CIFAR-100 veri seti üzerinde basit bir CNN mimarisi oluşturuldu ve eğitildi.
   - CNN, test kümesindeki görüntüler üzerinde tahminlerde bulunarak sonuçlar rapor edildi.

4. **Görselleştirme ve Değerlendirme**:
   - Her sınıflandırıcı için accuracy, precision, recall ve F1 score metrikleri görselleştirildi.
   - CNN tahminleri gerçek etiketlerle karşılaştırıldı ve sınıflandırma doğruluğu raporlandı.



**Kaggle Notebook Links:**
1. https://www.kaggle.com/code/codingwithmk/gaih-ml-supervised-project-cifar100-14-09-2024/edit
2. https://www.kaggle.com/code/codingwithmk/gaih-ml-unsupervised-project-cifar100-15-09-2024/edit