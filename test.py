def print_attributes(obj):
    """
    Prints all attributes and their values for a given object using vars().
    """
    # vars(obj) 返回对象的 __dict__ 属性（即包含所有属性的字典）
    # 题目要求的输出格式是：属性名 值
    for key, value in vars(obj).items():
        print(f"{key} {value}")


dict1 = {'a':1,'b':2,'c':3}

for key,v in dict1.items():
    print(f"{v}")
