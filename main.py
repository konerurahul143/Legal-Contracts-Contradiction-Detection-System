from src.contradiction_detector import ContradictionDetector

MODEL_PATH = "models/deberta_contractnli_best"

# Initialize detector
detector = ContradictionDetector(model_path=MODEL_PATH)

pdf_paths = [
    r"C:\Users\koner\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf\Part_I\Affiliate_Agreements\CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf",

    r"C:\Users\koner\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf\Part_I\Affiliate_Agreements\CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf",

    r"C:\Users\koner\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf\Part_I\Affiliate_Agreements\DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement.pdf"
]

# Step 1: Extract clauses
detector.process_documents(pdf_paths)

# Step 2: Build FAISS index
detector.build_vector_index()

# Step 3: Detect contradictions
contradictions = detector.detect_contradictions()

print(f"\nDetected {len(contradictions)} contradictions.")
print("=" * 100)

# Step 4: Display results
for i, c in enumerate(contradictions, start=1):

    print(f"\nContradiction {i}")
    print("=" * 100)

    print(f"Source Document : {c['source_document']}")
    print(f"Target Document : {c['target_document']}")
    print(f"Similarity      : {c['similarity']:.4f}")

    print("\nModel Prediction")
    print("-" * 60)
    print(f"Label       : {c['label']}")
    print(f"Confidence  : {c['confidence']:.4f}")

    print("\nSource Clause")
    print("-" * 60)
    print(c["source_text"])

    print("\nTarget Clause")
    print("-" * 60)
    print(c["target_text"])

    print("\nGemini Analysis")
    print("-" * 60)
    print(c["explanation"])

    print("\nNote")
    print("-" * 60)
    print(
        "The contradiction prediction above is produced by the fine-tuned "
        "DeBERTa ContractNLI model. "
        "Gemini is used only to summarize and compare the two clauses. "
        "It does not influence the prediction."
    )

print("\nFinished.")