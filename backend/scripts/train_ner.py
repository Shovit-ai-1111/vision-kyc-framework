import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
from app.evaluation.dataset import load_jsonl, validate_documents
from app.nlp.fields import LABELS, features


def train(args):
    rows = load_jsonl(args.manifest)
    counts = validate_documents(rows)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Training and validation documents are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.engine == "crf":
        import pycrfsuite

        trainer = pycrfsuite.Trainer(verbose=False)
        for r in training:
            trainer.append(
                [
                    features(r["tokens"], i, r["width"], r["height"])
                    for i in range(len(r["tokens"]))
                ],
                r["labels"],
            )
        trainer.set_params(
            {
                "c1": 0.05,
                "c2": 0.05,
                "max_iterations": 120,
                "feature.possible_transitions": True,
            }
        )
        trainer.train(str(output / "model.crfsuite"))
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, use_fast=True, add_prefix_space=True
        )
        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={x: i for i, x in enumerate(LABELS)},
            ignore_mismatched_sizes=True,
        ).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        best = float("inf")

        def windows(row):
            enc = tokenizer(
                [t["text"] for t in row["tokens"]],
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                stride=64,
                return_overflowing_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            for window in range(len(enc["input_ids"])):
                ids = enc.word_ids(window)
                previous = None
                labels = []
                for w in ids:
                    labels.append(
                        -100
                        if w is None or w == previous
                        else LABELS.index(row["labels"][w])
                    )
                    previous = w
                batch = {
                    k: v[window : window + 1].to(args.device)
                    for k, v in enc.items()
                    if k != "overflow_to_sample_mapping"
                }
                batch["labels"] = torch.tensor([labels], device=args.device)
                if model.config.model_type == "lilt":
                    batch["bbox"] = torch.tensor(
                        [
                            [
                                [
                                    int(
                                        v
                                        / (
                                            row["width"]
                                            if j % 2 == 0
                                            else row["height"]
                                        )
                                        * 1000
                                    )
                                    for j, v in enumerate(row["tokens"][w]["box"])
                                ]
                                if w is not None
                                else [0, 0, 0, 0]
                                for w in ids
                            ]
                        ],
                        device=args.device,
                    )
                yield batch

        for epoch in range(args.epochs):
            model.train()
            random.shuffle(training)
            for row in training:
                for batch in windows(row):
                    optimizer.zero_grad()
                    loss = model(**batch).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
            model.eval()
            losses = []
            with torch.inference_mode():
                for row in validation:
                    for batch in windows(row):
                        losses.append(float(model(**batch).loss))
            loss = float(np.mean(losses))
            print(f"Epoch {epoch + 1}: validation loss {loss:.5f}")
            if loss < best:
                best = loss
                model.save_pretrained(output, safe_serialization=True)
                tokenizer.save_pretrained(output)
    metadata = {
        "trained": True,
        "engine": args.engine,
        "base_model": args.base_model
        if args.engine == "transformer"
        else "CRFsuite linear-chain CRF",
        "data_kind": rows[0]["data_kind"],
        "labels": LABELS,
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "counts": counts,
        "seed": 42,
        "selection": "validation loss"
        if args.engine == "transformer"
        else "fixed hyperparameters; report held-out metrics separately",
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--engine", choices=["crf", "transformer"], default="crf")
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default="nielsr/lilt-xlm-roberta-base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=15)
    train(p.parse_args())
