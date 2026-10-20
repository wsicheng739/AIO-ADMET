class PropertyAdapter:
    name = "base"

    def build_model(self, vocab_size, norm, config):
        raise NotImplementedError

    def transform_output(self, logits):
        return logits

    def build_prediction_dataframe(self, items, ensemble, norm, args):
        raise NotImplementedError