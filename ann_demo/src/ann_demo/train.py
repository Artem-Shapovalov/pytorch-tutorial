import os, torch, csv
from ann_demo.data import scaler
from ann_demo.model import TinyMLP

def load_dataset(path, sc):
    f = open(path, "r", newline="")
    r = csv.reader(f)

    header = next(r)
    rows = list(r)
    f.close()

    features = []
    outputs = []

    for row in rows:
        vals = [float(v) for v in row]
        curr_features = vals[:-1]
        output = vals[-1]

        for i in range(len(header) - 1):
            curr_features[i] = sc.scale(header[i], curr_features[i])

        features.append(curr_features)
        outputs.append([output])

    return features, outputs

def accuracy(model, inputs, outputs):
    with torch.no_grad():
        actual = model(inputs)
        probabilities = torch.sigmoid(actual)
        predictions = (probabilities >= 0.5).float()
        return (predictions == outputs).float().mean().item()

def main():
    sc = scaler("data/coeffs.csv")

    training_features, training_outputs = load_dataset("data/train.csv", sc)
    validation_features, validation_outputs = load_dataset("data/val.csv", sc)

    training_features = torch.tensor(training_features, dtype=torch.float32)
    training_outputs = torch.tensor(training_outputs, dtype=torch.float32)
    validation_features = torch.tensor(validation_features, dtype=torch.float32)
    validation_outputs = torch.tensor(validation_outputs, dtype=torch.float32)

    model = TinyMLP();
    model.train(); # Switch model to the training mode
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        idx = torch.randperm(2000)
        training_features = training_features[idx]
        training_outputs = training_outputs[idx]

        i = 0; batch_count = 0;
        while i < 2000:
            x = training_features[i:i+100]
            y = training_outputs[i:i+100]
            i += 100; batch_count += 1;

            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    print("Accuracy: ", accuracy(model,
                                 validation_features, validation_outputs))

    # Prepare the place to save the model
    if os.path.exists("artifacts"):
        os.remove("artifacts/model.pt")
        os.rmdir("artifacts")
    os.mkdir("artifacts")

    # Save the model
    ckpt = { "model_state": model.state_dict() }
    torch.save(ckpt, "artifacts/model.pt")

if __name__ == "__main__":
    main()
