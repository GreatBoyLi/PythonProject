class LimitFieldElement:
    def __init__(self, num, order):
        """order必须是素数"""
        if num >= order or num < 0:
            err = f"元素{num}数值必须在0和{order - 1}之间"
            raise ValueError(err)
        self.num = num
        self.order = order

    def __repr__(self):
        return f"LimitFieldElement_{self.order}({self.num})"

    def __eq__(self, other):
        if other is None:
            return False
        return self.num == other.num and self.order == other.order

    def __ne__(self, other):
        if other is None:
            return True
        return self.num != other.num or self.order != other.order

    def __add__(self, other):
        if other is None:
            raise ValueError("元素不能是None")
        if self.order != other.order:
            raise TypeError("不能对两个不同有限域集合的元素做‘+’操作")
        num = (self.num + other.num) % self.order
        return self.__class__(num, self.order)

    def __mul__(self, other):
        if other is None:
            raise ValueError("元素不能是None")
        if self.order != other.order:
            raise TypeError("不能对两个不同有限域集合的元素做‘*’操作")
        num = (self.num * other.num) % self.order
        return self.__class__(num, self.order)

    def __pow__(self, power, modulo=None):
        while power < 0:
            power += self.order
        num = pow(self.num, power, self.order)
        return self.__class__(num, self.order)

    def __truediv__(self, other):
        if other is None:
            raise ValueError("元素不能是None")
        if self.order != other.order:
            raise TypeError("不能对两个不同有限域集合的元素做‘/’操作")
        # 费马小定理做除法操作
        negative = other ** (self.order - 2)
        num = (self.num * negative.num) % self.order
        return self.__class__(num, self.order)


if __name__ == '__main__':
    a = LimitFieldElement(3, 5)
    b = LimitFieldElement(4, 5)
    c = a * b
    print(c)
    e = c / b
    print(e)