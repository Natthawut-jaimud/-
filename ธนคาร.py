class BankAcconut():
    def __init__(self):
        self.__balance = 0

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"ฝากสำเร็จ {amount} บาท ")
        else :
            print("ยอดเงินฝากต้องมากกว่า 0")
    def withdraw(self, amount):
        if amount > 0 and amount <= self.__balance:
            self.__balance -= amount
            print(f'ถอนเงินสำเร็จ :  {amount} บาท')
        else:
            print(f'ยอดเงินไม่เพียงพอต่อการถอน {amount}') 
    def check_balance(self):
        return self.__balance
    

bankAcconut = BankAcconut()
bankAcconut.deposit(1000)
bankAcconut.withdraw(1000)
print(f"เช็คจำนวนเงิน {bankAcconut.check_balance()}")
