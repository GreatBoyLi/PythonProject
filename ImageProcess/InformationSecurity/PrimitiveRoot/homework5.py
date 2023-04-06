# 快速求 m^e mod n
def compute(m, e, n):
    a = e
    b = m
    c = 1
    count = 0
    print(f"初始值  a:{a}    b:{b}     c:{c}")
    while a != 0:
        if a % 2 == 0:
            a = a / 2
            b = (b * b) % n
        else:
            a = a - 1
            c = (c * b) % n
        count += 1
        print(f"{count}      a:{a}     b:{b}    c:{c}")
    return c


if __name__ == '__main__':
    d = compute(2008, 37, 77)
    print(d)
