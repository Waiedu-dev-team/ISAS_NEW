# Detailed Paper Outline - ISAS Human Activity Recognition

## 1. Introduction (Done)

### 1.1. Context and Motivation
Start by establishing the clinical importance of automated behavior monitoring for individuals with neurodevelopmental disabilities (NDD). Discuss the limitations and challenges of traditional manual observation, such as being time-consuming, subjective, and not scalable for 24/7 care. Cite sources on the prevalence and impact of challenging behaviors.

### 1.2. Technological Solutions and Paradigm Shift
Introduce Human Activity Recognition (HAR) as a key technological solution. Contrast traditional RGB video-based methods with skeleton-based approaches, highlighting the significant advantages of the latter in preserving privacy and its inherent robustness to environmental factors like lighting and background clutter.

### 1.3. Key Research Gaps
Clearly define the two primary challenges this paper addresses:
- **Generalization**: Discuss the critical issue of models failing to generalize to new, unseen subjects. Explain how standard evaluation protocols (like k-fold cross-validation) can be misleading and introduce the necessity of a stricter, more realistic protocol like Leave-One-Subject-Out (LOSO) Cross-Validation.
- **Fine-Grained Discrimination**: Highlight the persistent difficulty in distinguishing between activities with subtle and similar motion patterns (e.g., "sitting quietly" vs. "using phone").

### 1.4. Proposed Solution and Contributions
Introduce your work as a feature-driven framework that utilizes a robust ensemble classifier. State that the core of the methodology is a comprehensive feature engineering process followed by a rigorous model selection phase using the LOSO protocol to identify the most generalizable model. List the main contributions: 
1. A comprehensive multi-domain feature set for 2D pose data
2. A systematic evaluation demonstrating the superiority of the Extra Trees model for this subject-independent task
3. A hierarchical windowing strategy for applying the trained model to continuous, unlabeled data

### 1.5. Paper Structure
Conclude with a brief overview of the paper's organization.

## 2. Related Work

### 2.1. Skeleton-based Human Activity Recognition (Le Minh Hoang)
Provide a general overview of the field, discussing its evolution from early methods to current trends. Establish it as the core domain of your work.

### 2.2. Handcrafted Feature-based Methods (Thiên Ân)
Explain this approach in detail, as it forms the theoretical basis for your methodology. Discuss examples from the literature, such as kinematic features (velocity, acceleration) and geometric/relational features (inter-joint angles, distances).

### 2.3. Deep Learning-based Methods for HAR (Khoa Minh)
Retain this section to provide a complete academic context. Discuss RNNs, LSTMs, and Transformers. Use this section to argue why, in scenarios with limited training subjects and a high demand for generalization, a well-designed feature-based approach can be more robust and less prone to overfitting than data-hungry deep learning models.

### 2.4. Generalization and Evaluation in HAR (Qhuy)
Dedicate this subsection to the critical problem of model generalization. Explain in detail why traditional validation is often inadequate. Emphasize the importance of LOSO as a more realistic and rigorous protocol for assessing performance on new subjects, citing relevant literature that supports this claim.

## 3. Methodology

### 3.1. Overall Framework (Thiên Ân)
Present your new pipeline diagram as Figure 1. Verbally describe the diagram, explaining the flow of data through the main stages: Feature Engineering, Model Training, and the final Submission block which includes both LOSO Evaluation and prediction on the test file.

### 3.2. Dataset and Preprocessing (Thiên Ân)
#### Dataset Description
Describe the ISAS dataset from the challenge, detailing the 8 action classes (4 usual, 4 unusual) and the nature of the data (2D pose keypoints from 4 labeled participants).

#### Preprocessing and Segmentation
Detail the steps taken to clean the data (e.g., handling missing values). Explain your use of a sliding window approach (e.g., 120-frame window with 50% overlap) to segment the continuous time-series data into discrete samples for feature extraction.

### 3.3. Feature Engineering and Selection

#### Feature Extraction
Detail the specific categories of the 240 features you engineered. Organize them into the four primary groups:

##### Bounding Box Features (52 features) - (Khoa Minh - 1 paragraph ngắn)
Capturing holistic body movement.

##### Motion Features (62 features)
Describing individual joint kinematics.

##### Distance Features (81 features)
Encoding postural relationships.

##### Multi-domain Engineered Features (45 features)
For complex patterns like motion states and signal complexity.

#### Feature Selection (Lê Minh Hoàng)
Briefly explain the process used to reduce the dimensionality of the feature set and remove redundant or highly correlated features, leading to an optimized set for training. Explain ANOVA knowledge and how it's used in feature selection down to 240 features.

#### Standardization (Quốc Huy)
Mention the use of a technique like StandardScaler to normalize the feature values, ensuring that no single feature dominates the learning process due to its scale.

### 3.4. Model Training and Selection

#### Foundational Concepts of Ensemble Classifiers (Lê Minh Hoàng)
Briefly explain the principles behind Random Forest, and, most importantly, Extremely Randomized Trees (Extra Trees).

#### Highlight Extra Trees Advantages (Thiên Ân)
Highlight why the increased randomness of Extra Trees makes it a strong candidate for preventing overfitting and improving generalization.

#### Model Selection via LOSO (Khoa Minh)
Describe the systematic process of training and comparing several machine learning models (e.g., SVM, Random Forest, XGBoost, Extra Trees). Crucially, explain that this comparison was performed using the LOSO protocol, which led to the selection of Extra Trees as the most robust and generalizable model.

### 3.5. Prediction on Unseen Data (Hierarchical Windowing) - (Thiên Ân)
Explain the final step of your pipeline: applying the trained Extra Trees model to predict labels for a continuous, unlabeled data stream (e.g., the 5th participant's test file).

Describe the hierarchical windowing strategy used for this prediction task:
- Predictions are first made on short windows (e.g., 30 frames) to capture rapid, localized actions.
- Predictions are also made on long windows (e.g., 120 frames) to capture the broader context of sustained activities.
- Explain the weighted voting mechanism used to aggregate these multi-scale predictions into a single, final label for each frame, giving more weight to the predictions from longer, more context-rich windows.

## 4. Experiments and Results

### 4.1. Evaluation Protocol (Thiên Ân)
Formally state that all model comparison results are reported using the Leave-One-Subject-Out (LOSO) protocol. Define the performance metrics used: Accuracy, F1-Score, Precision, Recall, and the Confusion Matrix.

### 4.2. Model Performance Comparison (LOSO) - (Quốc Huy)
Present a clear table comparing the average LOSO performance of all the machine learning models you tested. This table should quantitatively demonstrate why Extra Trees was selected as the best model.

### 4.3. Analysis of the Best Model (Extra Trees) - (Thiên Ân)
- Present the detailed, aggregated confusion matrix for the Extra Trees model from the LOSO evaluation.
- Analyze the matrix to discuss which actions were classified correctly most often and which pairs of actions were most frequently confused, providing insight into the model's strengths and weaknesses.
- Present the results of a feature importance analysis from the trained Extra Trees model. List the top 10-15 most influential features and discuss what these features represent physically (e.g., "velocity of the right wrist," "aspect ratio of the body bounding box").

## 5. Discussion

### 5.1. Interpretation of Findings (Thiên Ân)
Analyze why the feature-driven approach with Extra Trees proved effective for this subject-independent task. Discuss the significance of the most important features—do they correspond to intuitive aspects of the actions? How does the model's performance on the LOSO evaluation reflect its potential for real-world application?

### 5.2. Comparison with Existing Work (Lê Minh Hoàng)
Relate your findings (both the performance and the important features) back to the papers discussed in the Related Work section.

### 5.3. Limitations (Khoa Minh)
Acknowledge the limitations of your study, such as the small number of subjects in the dataset, the use of simulated rather than clinical data, and any other constraints you faced.

## 6. Conclusion (Quốc Huy)

### 6.1. Summary of Contributions
Concisely summarize the problem, your proposed feature-driven methodology, and the key findings from your rigorous LOSO evaluation.

### 6.2. Final Remarks
Reiterate the main takeaway of your work—that a well-designed feature engineering pipeline combined with a suitable ensemble model offers a practical and generalizable solution for HAR in privacy-sensitive, data-limited contexts.

### 6.3. Future Work
Suggest concrete directions for future research, such as testing the framework on a larger, more diverse dataset; exploring real-time implementation; or investigating the fusion of this feature-based approach with lightweight deep learning models.

---

## Team Assignment Summary

### Le Minh Hoang
- Section 2.1: Skeleton-based Human Activity Recognition
- Section 3.3: Feature Selection (ANOVA explanation)
- Section 3.4: Foundational Concepts of Ensemble Classifiers
- Section 5.2: Comparison with Existing Work

### Thiên Ân
- Section 2.2: Handcrafted Feature-based Methods
- Section 3.1: Overall Framework
- Section 3.2: Dataset and Preprocessing
- Section 3.4: Extra Trees Advantages
- Section 3.5: Prediction on Unseen Data
- Section 4.1: Evaluation Protocol
- Section 4.3: Analysis of the Best Model
- Section 5.1: Interpretation of Findings

### Khoa Minh
- Section 2.3: Deep Learning-based Methods for HAR
- Section 3.3: Bounding Box Features (brief paragraph)
- Section 3.4: Model Selection via LOSO
- Section 5.3: Limitations

### Quốc Huy
- Section 2.4: Generalization and Evaluation in HAR
- Section 3.3: Standardization
- Section 4.2: Model Performance Comparison (LOSO)
- Section 6: Conclusion

---

**Note**: This outline focuses on the ISAS_BEST(2).ipynb notebook as the primary work, emphasizing the comprehensive feature engineering approach and LOSO evaluation methodology for real-world generalization. 