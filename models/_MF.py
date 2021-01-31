import numpy as np

"""
**Concept**
- 행렬분해 이용
- CF 기반, Matrix Factorization 계열의 알고리즘은 Model based CF
- CF의 user, item pivot_table에서 NaN값인것들을 행렬분해로 채워나감
- pivot_table 상의 rating(score)가 추론된 값으로 채워짐
<base code>
https://yamalab.tistory.com/92
> cost : RMSE
--> Gradient descent
"""

class MF():
    """
    Matrix Factorization
    """
    def __init__(self, table, latent, lr, reg_param, epochs, verbose = False):
        """
        :param rate: rating matrix
        :param latent: latent parameter
        :param lr: learning rate, alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: status
        """
        self._rate = np.nan_to_num(np.array([x for x in table.values()]))
        self._ids = [x for x in table.keys()]
        self._num_users, self._num_items = self._rate.shape
        self._latent = latent
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose
        self._lr = lr

    def fit(self):
        """
        Matrix Factorization 학습 : matrix latent weight, bias를 갱신
        -> cost를 최소로 만드는 q_i, p_i(?)학습
        :return:  training process(history)
        """
        # init latent features
        self._P = np.random.normal(size = (self._num_users, self._latent))
        self._Q = np.random.normal(size = (self._num_items, self._latent))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        # global bias : input R에서 평가된 rating의 평균값을 global bias로 사용한다.
        self._b = np.mean(self._rate[np.where(self._rate != 0)])
        # non zero values(rated)
        # self._b : 정규화 기능을 가진다. 최종 rating에 음수가 들어가는 것 대신 latent feature에
        # 음수가 포함되도록 해준다.

        # train while epochs
        self._training_process = [] # history

        for epoch in range(self._epochs):
            # rating이 존재하는 index를 기준으로 학습을 진행한다.
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._rate[i, j] > 0: # if rated
                        self.gradient_descent(i, j, self._rate[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch+1, cost))

    def get_complete_matrix(self):
        """
        compute matrix PXQ+P.bias+Q.bias+global_bias
        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을
                   해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_P[:, np.newaxis]+self._b_Q[np.newaxis:, ]+ self._P.dot(self._Q.T)

    def cost(self):
        """
        RMSE(Root Mean Squared Error)
        numpy.nonzero
        :return: rmse cost
        """
        xi, yi = self._rate.nonzero() # R[xi, yi], nonzero values
        predicted = self.get_complete_matrix()
        # init cost
        cost = 0
        for x, y in zip(xi, yi):
            # pow(a, b) : a의 b제곱을 계산해서 반환하는 함수
            # ex) pow(2, 10) : 1024
            cost += pow(self._rate[x, y] - predicted[x, y], 2)
            # adding -> sum of squared
        return np.sqrt(cost) / len(xi) # root, mean sse\

    def gradient(self, error, i, j):
        """
        Gradient of latent feature for GD
        :param error: rate - prediction
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple (user, item)
        """
        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq

    def gradient_descent(self, i, j, rating):
        """
        gradient descent function
        :param i: user index
        :param j: item index
        :param rating: rating of (i, j)
        """
        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._lr * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._lr * (error - self._reg_param * self._b_Q[j])

        # update latent features
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._lr * dp
        self._Q[j, :] += self._lr * dq

    def get_prediction(self, i, j):
        """self._ids = _table.items()
        get predicted rating : user_i, item_j
        :return : prediction of r_ij(hat(r_ij))
        """
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

    def print_results(self):
        """
        print fit results
        """

        print("User Latent P:")
        print(self._P)
        print("Item Latent Q:")
        print(self._Q.T)
        print("P x Q:")
        print(self._P.dot(self._Q.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_P)
        print("Item Latent bias:")
        print(self._b_Q)
        print("Final R matrix:") # Predicted rating
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs - 1][1])
    def estimated(self):
        return zip(self._ids, self.get_complete_matrix())

if __name__=="__main__":
    dict_rate = {0: [1, np.nan, np.nan, 1, 3],
                 1: [2, np.nan, 3, 1, 1],
                 2: [1, 2, 0, 5, 0],
                 3: [1, 0, 0, 4, 4],
                 4: [2, 1, 5, 4, 0],
                 5: [5, 1, 5, 4, 0],
                 6: [0, 0, 0, 1, 0], }
    factorizer = MF(dict_rate, latent=3, lr=0.01, reg_param=0.01, epochs=300, verbose=True)
    factorizer.fit()
    factorizer.print_results()
