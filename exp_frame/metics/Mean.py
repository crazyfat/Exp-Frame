class Mean:
    def __init__(self, name='mean'):
        self.name = name
        self._result = 0.0
        self._count = 0

    def update_state(self, values):
        self._count += len(values)
        self._result += (sum(values) - len(values) * self._result) / self._count

    def reset_state(self):
        self._count = 0
        self._result = 0.0

    def result(self):
        return self._result


if __name__ == '__main__':
    m = Mean()
    m.update_state([1, 3, 5, 7])
    print(m.result())
    m.update_state([5, 6, 7])
    print(m.result())
