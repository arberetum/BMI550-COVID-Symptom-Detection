from annotation import Annotation

class Post:
    def __init__(self, id, date, text):
        self._id = id
        self.date = date
        self.text = text

    def __eq__(self, other):
        return self._id == other.get_id()

    def __hash__(self):
        return hash(self._id)

    def get_id(self):
        return self._id


class AnnotatedPost(Post):
    def __init__(self, id, date, text, annotation):
        super().__init__(id, date, text)
        self.annotation = annotation

    def __init__(self, post, annotation):
        super().__init__(post.get_id(), post.date, post.text)
        self.annotation = annotation

    def to_dict(self):
        result = dict()
        result["ID"] = self.get_id()
        result["DATE"] = self.date
        result["TEXT"] = self.text
        annotation_dict = self.annotation.to_dict()
        result.update(annotation_dict)
        return result
