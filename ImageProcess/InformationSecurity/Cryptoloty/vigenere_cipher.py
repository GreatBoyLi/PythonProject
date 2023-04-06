# 加密
def encryption(string1, string2):
    res = ''
    cipherKey = list(string2)
    size = len(cipherKey)
    i = 0
    for x in string1:
        if 97 <= ord(x) <= 122:
            y = (ord(x) - 97 + (ord(cipherKey[i % size]) - 97)) % 26 + 65
            res += chr(y)
            i += 1
        else:
            res += x
    return res


# 解密
def decryption(string1, string2):
    res = ''
    cipherKey = list(string2)
    size = len(cipherKey)
    i = 0
    for x in string1:
        if 65 <= ord(x) <= 90:
            y = (ord(x) - 65 - (ord(cipherKey[i % size]) - 97)) % 26 + 97
            res += chr(y)
            i += 1
        else:
            res += x
    return res


if __name__ == '__main__':
    print('维吉尼亚密码')
    cipher = input('请输入密钥字母：')
    message = input('请输入明文：')
    ciphertext = encryption(message.lower(), cipher.lower())
    print(f'加密后的密文是：{ciphertext}')
    flag = input('是否需要解密？（Y：解密；N：不解密）')
    if flag == 'Y' or flag == 'y':
        plaintext = decryption(ciphertext, cipher.lower())
        print(f'解密后的明文是：{plaintext}')

