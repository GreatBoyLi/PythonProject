class EllipticPoint:
    def __init__(self, x, y, a, b):
        """
        y^2 = x^3 + a*x +b
        a = 0, b = 7 -> secp256k1
        'o'点，X=None或y=None
        """
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        # 判断零点
        if x is None or y is None:
            return
        # 判断点x, y是否在曲线上
        if (y ** 2) != (x ** 3 + a * x + b):
            raise ValueError(f"{x},{y}不在曲线上")

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.a == other.a and self.b == self.b

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y or self.a != other.a or self.b != self.b

    def __add__(self, other):
        """实现椭圆曲线的‘+’操作"""
        # 首先确定两点是否在同一个椭圆曲线上
        if self.a != other.a or self.b != other.b:
            raise TypeError(f"点{self},{other}不在同一条曲线上")

        # 如果有一个是0，加法的结果是直接返回另一个
        if self.x is None:
            return other
        if other.x is None:
            return self

        # 两点x相同，y互为相反数
        if self.x == other.x and self.y == -other.y:
            return self.__class__(None, None, self.a, self.b)

        # 两点连线与曲线交于第三点，需要使用韦达定理
        x1 = self.x
        y1 = self.y
        x2 = other.x
        y2 = other.y
        if self == other:
            # 切线、斜率需要使用微分
            k = (3 * self.x ** 2 + self.a) / 2 * self.y
        else:
            k = (y2 - y1) / (x2 - x1)
        # 韦达定理
        x3 = k ** 2 - x1 - x2
        y3 = k * (x1 - x3) - y1

        return self.__class__(x3, y3, self.a, self.b)

    def __repr__(self):
        return f"椭圆曲线:y\u00b2=x\u00b3+{self.a}x+{self.b},点是({self.x},{self.y})"


if __name__ == '__main__':
    a = EllipticPoint(None, None, 5, 7)
    b = EllipticPoint(-1, -1, 5, 7)
    c = EllipticPoint(-1, 1, 5, 7)
    d = EllipticPoint(2, 5, 5, 7)
    print(f"a+b:{a + b}")
    print(f"c+b:{c + b}")
    print(f"b+b:{b + b}")
    print(f"d+b:{d + b}")
    # print(a)
