import torch
from ann_demo.data import scaler
from ann_demo.model import TinyMLP

def main():
    model = TinyMLP()
    ckpt = torch.load("artifacts/model.pt")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    sc = scaler("data/coeffs.csv")

    inputs = [
            sc.scale("fails_24h", 3.0),
            sc.scale("fails_10m", 1.0),
            sc.scale("account_age_years", 0.8),
            sc.scale("ip_rep", 0.2),
            sc.scale("geo_km", 1500.0),
            sc.scale("new_device", 1.0),
            sc.scale("time_anom", 0.6),
            sc.scale("pwd_change_7d", 0.0),
            sc.scale("velocity_h", 0.3),
            sc.scale("mfa_enabled", 1.0)
            ]

    outputs = []
    with torch.no_grad():
        itensor = torch.tensor(inputs, dtype=torch.float32)
        outputs = torch.sigmoid(model(itensor)).tolist()

    print("Outputs: ", outputs)
    if outputs[0] < 0.3:
        print("ALLOW")
    elif outputs[0] < 0.7:
        print("CHALLENGE")
    else:
        print("BLOCK")

if __name__ == "__main__":
    main()
