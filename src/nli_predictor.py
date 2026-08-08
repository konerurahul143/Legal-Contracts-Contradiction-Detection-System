import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIPredictor:
    """
    Natural Language Inference Predictor.

    Labels
    ------
    0 -> NotMentioned
    1 -> Entailment
    2 -> Contradiction
    """

    def __init__(
        self,
        model_path,
        max_length=512,
        device=None
    ):

        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"Loading tokenizer from Microsoft DeBERTa-v3-base...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-base"
        )

        print(f"Loading fine-tuned model from: {model_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)

        self.model.eval()

        self.id2label = {
            0: "NotMentioned",
            1: "Entailment",
            2: "Contradiction"
        }

        print(f"Model loaded successfully on {self.device}")

    ####################################################################
    # Batch Prediction
    ####################################################################

    def predict_batch(
        self,
        premises,
        hypotheses
    ):

        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = F.softmax(
                outputs.logits,
                dim=1
            )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        results = []

        for prediction, probability in zip(
            predictions,
            probabilities
        ):

            prediction = prediction.item()

            results.append({

                "label": self.id2label[prediction],

                "confidence": float(
                    probability[prediction]
                ),

                "probabilities": {

                    self.id2label[i]: float(
                        probability[i]
                    )

                    for i in range(3)
                }
            })

        return results

    ####################################################################
    # Single Prediction
    ####################################################################

    def predict(
        self,
        premise,
        hypothesis
    ):

        return self.predict_batch(
            [premise],
            [hypothesis]
        )[0]