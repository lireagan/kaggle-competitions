#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = "reagan.llx"

import os
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append("{}/src".format(PROJECT_DIR))

from util.visualization_util import plot_2d_decision_regions


def main():
    # step1: 数据集获取
    labeled_data = pd.read_csv("{}/data/titanic/train.csv".format(PROJECT_DIR))
    # test_data = pd.read_csv("{}/data/titanic/test.csv".format(PROJECT_DIR))
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(labeled_data[features])  # get_dummies是进行one-hot处理
    y = labeled_data["Survived"]

    # step2: 数据集切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)  # train_test_split方法分割数据集

    # step3: 特征标准化
    sc = StandardScaler()
    sc.fit(X_train)  # 使用训练集特征来设置z-score的参数（均值和方差）
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # step4: 降维到2d以便可视化
    pca = PCA(n_components=2)
    pca.fit(X_train_std)  # 使用训练集特征来设置PCA的参数
    X_train_pca = pca.transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # step5: 构造多个分类器
    classifiers = {
        "KNN": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel="linear", C=0.025),
        "RBF SVM": SVC(gamma=2, C=1),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "RF": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    # step6: 执行决策边界可视化
    X_combined_pca = np.vstack((X_train_pca, X_test_pca))  # (sample_size, 2)
    y_combined = np.hstack((y_train, y_test))  # (sample_size, )

    plot_2d_decision_regions(X=X_combined_pca, y=y_combined, classifiers=classifiers,
                             test_idx=range(X_train_pca.shape[0], X_combined_pca.shape[0]))


if __name__ == '__main__':
    main()
