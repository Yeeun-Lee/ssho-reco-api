import numpy as np

def cos_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def centroidVec(vec, mode = "mean"):
    """
    최근 10개 아이템에 대한 대표 벡터 생성 로직
    -> 벡터 공간에서 10개 아이템 벡터들의 중심(centroid)

    :param vec: 10 latest vectors(np.ndarray)
    :param mode: mean, median
    :return: n-dimensional centroid vector
    """
    if mode == "median":
        return np.median(vec)
    else:
        return np.mean(vec)