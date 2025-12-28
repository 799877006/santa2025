class DNA:
    def __init__(self, sequence):
        self.sequence = sequence


    # def __str__(self):
    #     return self.sequence

    def __add__(self,other):
        return self.sequence + other.sequence

dna1 = DNA("ATCG")
dna2 = DNA("ATCG")
print(dna1 + dna2)


print(dna1)