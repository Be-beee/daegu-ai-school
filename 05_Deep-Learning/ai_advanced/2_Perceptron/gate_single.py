# 단층 퍼셉트론 이용하여 AND NAND OR 게이트 구현

# AND
# 두개의 입력 값이 모두 1인 경우만 output 1 아니면 0

def AND_gate(x1, x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.7

    result = x1 * w1 + x2 * w2 + b # 단층 퍼셉트론 공식

    if result <= 0:
        return 0
    else:
        return 1



print(AND_gate(0, 0), AND_gate(0, 1), AND_gate(1, 0), AND_gate(1, 1))



# NAND
# 두개의 입력값이 모두 1인 경우에만 output 1 아니면 0

def NAND_gate(x1, x2):
    w1 = -0.5
    w2 = -.5
    b = 0.7

    result = x1 * w1 + x2 * w2 + b

    if result <= 0:
        return 0
    else:
        return 1


print(NAND_gate(0, 0), NAND_gate(0, 1), NAND_gate(1, 0), NAND_gate(1, 1))




# OR
# 두 값이 0 0 -> 0, 1 -> 1 두 값이 서로 다르면 1

def OR_gate(x1, x2):
    w1 = .5
    w2 = .5
    b = -.4

    result = x1 * w1 + x2 * w2 + b

    if result <= 0:
        return 0
    else:
        return 1


print(OR_gate(0, 0), OR_gate(0, 1), OR_gate(1, 0), OR_gate(1, 1))