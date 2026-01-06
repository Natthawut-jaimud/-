width = int(input('กว้าง'))
hight = int(input('ยาว'))

def area(width,hight):
    reg = width * hight
    tri = (1/2) * width * hight
    print(f'reg = {reg} and tri = {tri}')
    return (reg,tri)


rec1,tri1 = area(width,hight)
print(f'พื่นที่สีเหลียม {rec1}')
print(f"พื้นที่สามเหลียม {tri1}")
