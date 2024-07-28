import numpy as np


def feature_normalize(features):
    theMax = -100000
    theMin = 100000
    for i in range(len(features)):
        for j in range(len(features[i])):
            for k in range(len(features[i][j])):
                if features[i][j][k] < theMin:
                    theMin = features[i][j][k]
                if features[i][j][k] > theMax:
                    theMax = features[i][j][k]
    for i in range(len(features)):
        for j in range(len(features[i])):
            for k in range(len(features[i][j])):
                features[i][j][k] = (features[i][j][k] - theMin) / (theMax - theMin)
    return features
