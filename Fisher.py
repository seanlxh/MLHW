# -*- coding:utf-8 -*-
from sklearn.datasets import make_multilabel_classification
import numpy as np
import warnings
class FLD(object):
    def __init__(self,c_1,c_2):
        if(c_1.shape[0] == 0 | c_2.shape[0] == 0):
            warnings.warn("长度为0")
        self.c_1 = c_1
        self.c_2 = c_2
        self.w = c_1.shape[1]

    def cal_cov_and_avg(self,samples):
        """
        给定一个类别的数据，计算协方差矩阵和平均向量
        :param samples:
        :return:
        """
        u1 = np.mean(samples, axis=0)
        cov_m = np.zeros((samples.shape[1], samples.shape[1]))
        for s in samples:
            t = s - u1
            cov_m += t * t.reshape(t.shape[0], 1)
        return cov_m, u1

    def fit(self, X=None, y=None, sample_weight=None):
        """
        fisher算法实现
        :param c_1:
        :param c_2:
        :return:
        """
        cov_1, u1 = self.cal_cov_and_avg(self.c_1)
        cov_2, u2 = self.cal_cov_and_avg(self.c_2)
        s_w = cov_1 + cov_2
        u, s, v = np.linalg.svd(s_w)  # 奇异值分解
        s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
        self.w = np.dot(s_w_inv, u1 - u2)
        return self.w

    def judge(self,sample):
        """
        true 属于1
        false 属于2
        :param sample:
        :param w:
        :param center_1:
        :param center_2:
        :return:
        """
        u1 = np.mean(self.c_1, axis=0)
        u2 = np.mean(self.c_2, axis=0)
        center_1 = np.dot(self.w.T, u1)
        center_2 = np.dot(self.w.T, u2)
        pos = np.dot(self.w.T, sample)
        return abs(pos - center_1) < abs(pos - center_2)

