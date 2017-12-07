class RandomTranslate(object):
    def __init__(self, std):
        self.std = abs(std)

    def __call__(self, data):
        position = data.position
        translate = position.new(position.size()).uniform_(-self.std, self.std)
        data.position = position + translate
        return data
