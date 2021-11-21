import numpy as np


class BAM():
    def __init__(self, data):
        self.data_bipolar = []

        for item in data:
            self.data_bipolar.append(
                            [self.make_bipolar(item[0]),
                             self.make_bipolar(item[1])]
            )

        self.len_x = len(self.data_bipolar[0][1])
        self.len_y = len(self.data_bipolar[0][0])

        self.M = np.zeros([self.len_y, self.len_x])

        self.create_matrix()

    def create_matrix(self):
        for pair in self.data_bipolar:
            X = pair[0]
            Y = pair[1]
            for idx, xi in enumerate(X):
                for idy, yi in enumerate(Y):
                    self.M[idx][idy] += xi * yi
    
    def get_matrix(self):
        return self.M

    def multiply_by_vec(self, vec):
        return np.matmul(vec, self.M)

    def get_association(self, A):
        A = self.multiply_by_vec(A)
        return self.make_binary(A)

    def make_binary(self, vec):
        binary_vec = []
        for element in vec:
            if element <0:
                binary_vec = np.append(binary_vec, 0)
            else:
                binary_vec = np.append(binary_vec, 1)
        return binary_vec

    def make_bipolar(self, vec):
        bipolar_vec = np.array([])
        for element in vec:
            if element == 0:
                bipolar_vec = np.append(bipolar_vec, -1)
            else:
                bipolar_vec = np.append(bipolar_vec, 1)
        return bipolar_vec


'''
if __name__ == "__main__":
    data = [
    [[1,0,0,0], [1,0,0,0]],
    [[0,1,0,0], [1,1,0,0]],
    [[0,0,1,0], [1,1,1,0]],
    [[0,0,0,1], [1,1,1,0]]
    ]

    b = BAM(data)
    print(b.get_association([1,0,0,0]))
    print(b.get_association([0,1,0,0]))
    print(b.get_association([0,0,1,0]))
    print(b.get_association([0,0,0,1]))
'''
if __name__ == "__main__":
    data_pairs  = [
        [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0]],
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0]]
        ]
    b = BAM(data_pairs)

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    print('Matrix: ')
    pp.pprint(b.get_matrix())
    print('\n')
    print('[1, 0, 1, 0, 1, 0] ---> ', b.get_association([1, 0, 1, 0, 1, 0]))
    print('[1, 1, 1, 0, 0, 0] ---> ', b.get_association([1, 1, 1, 0, 0, 0]))