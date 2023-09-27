class SymptomExpression:
    def __init__(self, standard_symptom, cui, expression, negated):
        self.standard_symptom = str(standard_symptom)
        self.cui = str(cui)
        self.expression = str(expression)
        self.negated = str(negated)

    def __eq__(self, other):
        return self.expression == other.expression

    def __str__(self):
        return f"{self.standard_symptom}\t{self.cui}\t{self.expression}"
