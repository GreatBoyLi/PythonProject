def mod_fraction(a, b, p):
    """
    使用费马小定理求 a/b 模 p 的值，其中 p 为质数
    """
    # 计算 b^(p-2) 模 p 的值，即 b 在模 p 意义下的逆元
    b_inv = pow(b, p-2, p)
    # 计算 a * b_inv 的值，并对 p 取模
    return (a * b_inv) % p


class EBBC:
    def __init__(self, x, y, a, b, p=23):
        """
        y^2 = x^3 + a*x +b
        a = 0, b = 7 -> secp256k1
        'o'点，X=None或y=None
        """
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.p = p
        # 判断零点
        if x is None or y is None:
            return
        # 判断点x, y是否在曲线上
        a1 = y ** 2 % 23
        a2 = (x ** 3 + a * x + b) % 23
        if (y ** 2) % self.p != (x ** 3 + a * x + b) % self.p:
            raise ValueError(f"{x},{y}不在曲线上")

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.a == other.a and self.b == self.b and self.p == other.p

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y or self.a != other.a or self.b != self.b or self.p != other.p

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
            return self.__class__(None, None, self.a, self.b, self.p)

        # 两点连线与曲线交于第三点，需要使用韦达定理
        x1 = self.x
        y1 = self.y
        x2 = other.x
        y2 = other.y
        if self == other:
            # 切线、斜率需要使用微分
            k = mod_fraction(3 * self.x ** 2 + self.a,  2 * self.y, self.p)
            print(f"k=3x\u2081\u00b2 + a / 2y\u2081。"
                  f"求分数：{2 * self.y}分之{3 * self.x ** 2 + self.a}"
                  f"({3 * self.x ** 2 + self.a}/{2 * self.y})模{self.p}的余数，是：{k}")
        else:
            k = mod_fraction(y2 - y1, (x2 - x1), self.p)
            print(
                f"k=y\u2082 - y\u2081 / x\u2082 - x\u2081。"
                f"求分数：{y2 - y1}分之{x2 - x1}"
                f"({x2 - x1}/{y2 - y1})模{self.p}的余数，是：{k}")
        # 韦达定理
        x3 = (k ** 2 - x1 - x2) % self.p
        y3 = (k * (x1 - x3) - y1) % self.p

        return self.__class__(x3, y3, self.a, self.b, self.p)

    def __mul__(self, other):
        result = self
        for i in range(2, other+1):
            result += self
        return result

    def __repr__(self):
        return f"椭圆曲线:y\u00b2=x\u00b3+{self.a}x+{self.b},点是({self.x},{self.y})，有限域是{self.p}"


if __name__ == '__main__':
    # 椭圆曲线是 y^2 = x^3 + x + 1, 基点是A(0,1)
    A = EBBC(0, 1, 1, 1)
    n = 2
    print(f"{A}, 此点乘以{n},即{n}A的结果是:{A * n}")
