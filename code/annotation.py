from symptom import SymptomExpression


class Annotation:
    def __init__(self, post_id, expressions, standard_symptoms, cuis, negations):
        self.post_id = post_id
        self.symptom_count = len(expressions)
        if (len(standard_symptoms) != self.symptom_count) or (len(cuis) != self.symptom_count) or \
            (len(negations) != self.symptom_count):
            raise RuntimeError("Numbers of symptom expressions, standard symptoms, CUIs, and negations do not match")
        self.symptoms = list()
        for i in range(self.symptom_count):
            self.symptoms.append(SymptomExpression(standard_symptoms[i], cuis[i], expressions[i], negations[i]))

    def to_dict(self):
        annotation_dict = dict()
        expressions = [symptom.expression for symptom in self.symptoms]
        standard_symptoms = [symptom.standard_symptom for symptom in self.symptoms]
        cuis = [symptom.cui for symptom in self.symptoms]
        negations = [symptom.negated for symptom in self.symptoms]
        annotation_dict["Symptom Expressions"] = "$$$" + "$$$".join(expressions) + "$$$"
        annotation_dict["Standard Symptom"] = "$$$" + "$$$".join(standard_symptoms) + "$$$"
        annotation_dict["Symptom CUIs"] = "$$$" + "$$$".join(cuis) + "$$$"
        annotation_dict["Negation Flag"] = "$$$" + "$$$".join(negations) + "$$$"
        return annotation_dict