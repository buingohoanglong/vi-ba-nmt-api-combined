class ModelTypes:
    TRANSFORMER = 'Transformer'
    PHOBERT_FUSED = 'PhoBERT-fused NMT'
    LOAN_FORMER = 'Loanformer'
    BART_PHO = "BartPho"
    COMBINED = "Combined"

    @classmethod
    def get_models(cls):
        return [
            ModelTypes.COMBINED,
            ModelTypes.LOAN_FORMER,
            ModelTypes.PHOBERT_FUSED,
            ModelTypes.TRANSFORMER,
            ModelTypes.BART_PHO
        ]
