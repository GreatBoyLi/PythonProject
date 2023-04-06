def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1
        print('a')


a = count_up_to(5)
for b in a:
    print(b)
